"""
VoxelNeXt sparse backbone (Chen et al., CVPR 2023), ported from OpenPCDet
(pcdet/models/backbones_3d/spconv_backbone_voxelnext.py, Apache-2.0) with no change to the architecture.

Six sparse stages (strides 1..32). Stages 5 and 6 are moved back onto the stride-8 grid, concatenated with
stage 4, and the height axis is summed away, so the output is a *sparse* 2D BEV tensor at stride 8
(`encoded_spconv_tensor`) that VoxelNeXtHead predicts boxes from directly: no dense BEV map, no anchors, no
second stage.
"""
from functools import partial

import torch
import torch.nn as nn

import spconv.pytorch as spconv


def post_act_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,
                   conv_type='subm', norm_fn=None):
    if conv_type == 'subm':
        conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
    elif conv_type == 'spconv':
        conv = spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                   bias=False, indice_key=indice_key)
    elif conv_type == 'inverseconv':
        conv = spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False)
    else:
        raise NotImplementedError(conv_type)
    return spconv.SparseSequential(conv, norm_fn(out_channels), nn.ReLU())


class SparseBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, norm_fn=None, downsample=None, indice_key=None):
        super().__init__()
        assert norm_fn is not None
        bias = norm_fn is not None
        self.conv1 = spconv.SubMConv3d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=bias,
                                       indice_key=indice_key)
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()
        self.conv2 = spconv.SubMConv3d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias,
                                       indice_key=indice_key)
        self.bn2 = norm_fn(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = out.replace_feature(self.relu(self.bn1(out.features)))
        out = self.conv2(out)
        out = out.replace_feature(self.bn2(out.features))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out.replace_feature(self.relu(out.features + identity.features))
        return out


class VoxelResBackBone8xVoxelNeXt(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        k = model_cfg.get('SPCONV_KERNEL_SIZES', [3, 3, 3, 3])
        channels = model_cfg.get('CHANNELS', [16, 32, 64, 128, 128])
        out_channel = model_cfg.get('OUT_CHANNEL', 128)

        # (z, y, x); one extra z cell as in OpenPCDet
        self.sparse_shape = list(grid_size[::-1] + [1, 0, 0])

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, channels[0], 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(channels[0]),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            SparseBasicBlock(channels[0], channels[0], norm_fn=norm_fn, indice_key='res1'),
            SparseBasicBlock(channels[0], channels[0], norm_fn=norm_fn, indice_key='res1'),
        )
        self.conv2 = spconv.SparseSequential(
            block(channels[0], channels[1], k[0], norm_fn=norm_fn, stride=2, padding=int(k[0] // 2),
                  indice_key='spconv2', conv_type='spconv'),
            SparseBasicBlock(channels[1], channels[1], norm_fn=norm_fn, indice_key='res2'),
            SparseBasicBlock(channels[1], channels[1], norm_fn=norm_fn, indice_key='res2'),
        )
        self.conv3 = spconv.SparseSequential(
            block(channels[1], channels[2], k[1], norm_fn=norm_fn, stride=2, padding=int(k[1] // 2),
                  indice_key='spconv3', conv_type='spconv'),
            SparseBasicBlock(channels[2], channels[2], norm_fn=norm_fn, indice_key='res3'),
            SparseBasicBlock(channels[2], channels[2], norm_fn=norm_fn, indice_key='res3'),
        )
        self.conv4 = spconv.SparseSequential(
            block(channels[2], channels[3], k[2], norm_fn=norm_fn, stride=2, padding=int(k[2] // 2),
                  indice_key='spconv4', conv_type='spconv'),
            SparseBasicBlock(channels[3], channels[3], norm_fn=norm_fn, indice_key='res4'),
            SparseBasicBlock(channels[3], channels[3], norm_fn=norm_fn, indice_key='res4'),
        )
        self.conv5 = spconv.SparseSequential(
            block(channels[3], channels[4], k[3], norm_fn=norm_fn, stride=2, padding=int(k[3] // 2),
                  indice_key='spconv5', conv_type='spconv'),
            SparseBasicBlock(channels[4], channels[4], norm_fn=norm_fn, indice_key='res5'),
            SparseBasicBlock(channels[4], channels[4], norm_fn=norm_fn, indice_key='res5'),
        )
        self.conv6 = spconv.SparseSequential(
            block(channels[4], channels[4], k[3], norm_fn=norm_fn, stride=2, padding=int(k[3] // 2),
                  indice_key='spconv6', conv_type='spconv'),
            SparseBasicBlock(channels[4], channels[4], norm_fn=norm_fn, indice_key='res6'),
            SparseBasicBlock(channels[4], channels[4], norm_fn=norm_fn, indice_key='res6'),
        )
        self.conv_out = spconv.SparseSequential(
            spconv.SparseConv2d(channels[3], out_channel, 3, stride=1, padding=1, bias=False,
                                indice_key='spconv_down2'),
            norm_fn(out_channel),
            nn.ReLU(),
        )
        self.shared_conv = spconv.SparseSequential(
            spconv.SubMConv2d(out_channel, out_channel, 3, stride=1, padding=1, bias=True),
            nn.BatchNorm1d(out_channel),
            nn.ReLU(True),
        )

        self.num_point_features = out_channel
        self.backbone_channels = {
            'x_conv1': channels[0], 'x_conv2': channels[1], 'x_conv3': channels[2], 'x_conv4': channels[3],
        }

    @staticmethod
    def bev_out(x_conv):
        """Sum features over z for every (batch, y, x): sparse 3D -> sparse 2D."""
        features_cat = x_conv.features
        indices_cat = x_conv.indices[:, [0, 2, 3]]
        spatial_shape = x_conv.spatial_shape[1:]

        indices_unique, _inv = torch.unique(indices_cat, dim=0, return_inverse=True)
        features_unique = features_cat.new_zeros((indices_unique.shape[0], features_cat.shape[1]))
        features_unique.index_add_(0, _inv, features_cat)

        return spconv.SparseConvTensor(
            features=features_unique, indices=indices_unique,
            spatial_shape=spatial_shape, batch_size=x_conv.batch_size,
        )

    def forward(self, batch_dict):
        """
        batch_dict: voxel_features (num_voxels, C), voxel_coords (num_voxels, 4) [batch_idx, z, y, x], batch_size
        adds: encoded_spconv_tensor (sparse 2D, stride 8), encoded_spconv_tensor_stride, multi_scale_3d_features
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features, indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape, batch_size=batch_dict['batch_size'],
        )
        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)
        x_conv5 = self.conv5(x_conv4)
        x_conv6 = self.conv6(x_conv5)

        # put stride-16 and stride-32 voxels back on the stride-8 grid and merge them with stage 4
        x_conv5.indices[:, 1:] *= 2
        x_conv6.indices[:, 1:] *= 4
        x_conv4 = x_conv4.replace_feature(torch.cat([x_conv4.features, x_conv5.features, x_conv6.features]))
        x_conv4.indices = torch.cat([x_conv4.indices, x_conv5.indices, x_conv6.indices])

        out = self.bev_out(x_conv4)
        out = self.conv_out(out)
        out = self.shared_conv(out)

        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8,
            'multi_scale_3d_features': {
                'x_conv1': x_conv1, 'x_conv2': x_conv2, 'x_conv3': x_conv3, 'x_conv4': x_conv4,
            },
            'multi_scale_3d_strides': {'x_conv1': 1, 'x_conv2': 2, 'x_conv3': 4, 'x_conv4': 8},
        })
        return batch_dict
