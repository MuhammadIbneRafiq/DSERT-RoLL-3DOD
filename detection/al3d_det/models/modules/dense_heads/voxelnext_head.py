"""
VoxelNeXt sparse head (Chen et al., CVPR 2023), ported from OpenPCDet
(pcdet/models/dense_heads/voxelnext_head.py, Apache-2.0).

Changes from OpenPCDet, none of which change what the model computes:
- gt_boxes are cut to (x, y, z, dx, dy, dz, heading, class) before target assignment, as CenterHead in this repo
  does: DSERT-RoLL labels carry two extra (all-zero) velocity columns that the head has no output for.
- The IoU-branch losses compute the IoU between each target box and its own prediction with boxes_iou3d_gpu(...)
  .diagonal(), because this repo's compiled iou3d_nms op has no boxes_aligned_overlap_bev_gpu. Same values, no
  rebuild of the CUDA ops needed. Prediction/target pairs are taken in the same order the targets were written.
- Double-flip test-time augmentation is not ported (OpenPCDet uses it for nuScenes only).
- At inference the IoU-branch value (iou + 1) / 2 is clamped to [0, 1] before hm^(1-r) * iou^r; OpenPCDet does not
  clamp, so an output below -1 there gives a NaN score. Values inside [0, 1] are unchanged.
"""
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict
from torch.nn.init import kaiming_normal_

import spconv.pytorch as spconv

from al3d_utils.ops.iou3d_nms import iou3d_nms_utils
from al3d_det.utils import centernet_utils, loss_utils, model_nms_utils


class SeparateHead(nn.Module):
    def __init__(self, input_channels, sep_head_dict, kernel_size, init_bias=-2.19, use_bias=False):
        super().__init__()
        self.sep_head_dict = sep_head_dict

        for cur_name in self.sep_head_dict:
            output_channels = self.sep_head_dict[cur_name]['out_channels']
            num_conv = self.sep_head_dict[cur_name]['num_conv']

            fc_list = []
            for _ in range(num_conv - 1):
                fc_list.append(spconv.SparseSequential(
                    spconv.SubMConv2d(input_channels, input_channels, kernel_size, padding=int(kernel_size // 2),
                                      bias=use_bias, indice_key=cur_name),
                    nn.BatchNorm1d(input_channels),
                    nn.ReLU(),
                ))
            fc_list.append(spconv.SubMConv2d(input_channels, output_channels, 1, bias=True,
                                             indice_key=cur_name + 'out'))
            fc = nn.Sequential(*fc_list)
            if 'hm' in cur_name:
                fc[-1].bias.data.fill_(init_bias)
            else:
                for m in fc.modules():
                    if isinstance(m, spconv.SubMConv2d):
                        kaiming_normal_(m.weight.data)
                        if hasattr(m, 'bias') and m.bias is not None:
                            nn.init.constant_(m.bias, 0)
            self.__setattr__(cur_name, fc)

    def forward(self, x):
        return {cur_name: self.__getattr__(cur_name)(x).features for cur_name in self.sep_head_dict}


def _matched_pairs(pred, mask, ind, box_gt, batch_index):
    """
    For each frame: the predictions at the target voxels of the GT boxes that got a target (mask = 1), and those
    GT boxes, in the same order. Target k of a frame was written from GT box k, and mask is 1 only for k < N.
    """
    preds, gts = [], []
    for bs_idx in range(mask.shape[0]):
        m = mask[bs_idx]
        gt = box_gt[bs_idx]
        n = min(gt.shape[0], m.shape[0])
        preds.append(pred[batch_index == bs_idx][ind[bs_idx]][m])
        gts.append(gt[:n][m[:n]])
    return preds, gts


def aligned_iou3d(boxes_a, boxes_b):
    """IoU of boxes_a[i] with boxes_b[i] (both (N, 7)); what OpenPCDet's boxes_aligned_iou3d_gpu returns."""
    if boxes_a.shape[0] == 0:
        return boxes_a.new_zeros((0, 1))
    iou = iou3d_nms_utils.boxes_iou3d_gpu(boxes_a[:, :7].float().contiguous(), boxes_b[:, :7].float().contiguous())
    return iou.diagonal().unsqueeze(-1)


class IouLossSparse(nn.Module):
    """L1 between the IoU branch output (in [-1, 1]) and 2 * IoU(prediction, target box) - 1."""

    def forward(self, iou_pred, mask, ind, box_pred, box_gt, batch_index):
        if mask.sum() == 0:
            return iou_pred.new_zeros((1,)).sum()
        mask = mask.bool()
        iou_preds, _ = _matched_pairs(iou_pred, mask, ind, box_gt, batch_index)
        box_preds, gts = _matched_pairs(box_pred, mask, ind, box_gt, batch_index)
        loss = 0
        for p, b, g in zip(iou_preds, box_preds, gts):
            target = 2 * aligned_iou3d(b, g) - 1
            loss = loss + F.l1_loss(p, target, reduction='sum')
        return loss / (mask.sum() + 1e-4)


class IouRegLossSparse(loss_utils.IouRegLossSparse):
    """Distance-IoU loss on the decoded boxes (OpenPCDet's), with prediction/target pairing as above."""

    def forward(self, box_pred, mask, ind, box_gt, batch_index):
        if mask.sum() == 0:
            return box_pred.new_zeros((1,)).sum()
        mask = mask.bool()
        box_preds, gts = _matched_pairs(box_pred, mask, ind, box_gt, batch_index)
        loss = 0
        for b, g in zip(box_preds, gts):
            if b.shape[0]:
                loss = loss + (1. - self.bbox3d_iou_func(b, g)).sum()
        return loss / (mask.sum() + 1e-4)


class VoxelNeXtHead(nn.Module):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range, voxel_size,
                 predict_boxes_when_training=False):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class
        self.grid_size = grid_size
        self.point_cloud_range = torch.Tensor(point_cloud_range).cuda()
        self.voxel_size = torch.Tensor(voxel_size).cuda()
        self.feature_map_stride = self.model_cfg.TARGET_ASSIGNER_CONFIG.get('FEATURE_MAP_STRIDE', None)

        self.class_names = class_names
        self.class_names_each_head = []
        self.class_id_mapping_each_head = []
        self.gaussian_ratio = self.model_cfg.get('GAUSSIAN_RATIO', 1)
        self.gaussian_type = self.model_cfg.get('GAUSSIAN_TYPE', ['nearst', 'gt_center'])
        # IoU branch: OpenPCDet's Waymo setting (score = hm^(1-r) * iou^r, then class-wise NMS)
        self.iou_branch = self.model_cfg.get('IOU_BRANCH', False)
        if self.iou_branch:
            self.rectifier = list(self.model_cfg.get('RECTIFIER'))
            nms_configs = self.model_cfg.POST_PROCESSING.NMS_CONFIG
            self.nms_configs = [EasyDict(NMS_TYPE=nms_configs.NMS_TYPE,
                                         NMS_THRESH=nms_configs.NMS_THRESH[i],
                                         NMS_PRE_MAXSIZE=nms_configs.NMS_PRE_MAXSIZE[i],
                                         NMS_POST_MAXSIZE=nms_configs.NMS_POST_MAXSIZE[i]) for i in range(num_class)]
        assert not self.model_cfg.get('DOUBLE_FLIP', False), 'DOUBLE_FLIP is not ported'

        for cur_class_names in self.model_cfg.CLASS_NAMES_EACH_HEAD:
            self.class_names_each_head.append([x for x in cur_class_names if x in class_names])
            cur_class_id_mapping = torch.from_numpy(np.array(
                [self.class_names.index(x) for x in cur_class_names if x in class_names]
            )).cuda()
            self.class_id_mapping_each_head.append(cur_class_id_mapping)

        total_classes = sum([len(x) for x in self.class_names_each_head])
        assert total_classes == len(self.class_names), f'class_names_each_head={self.class_names_each_head}'

        kernel_size_head = self.model_cfg.get('KERNEL_SIZE_HEAD', 3)

        self.heads_list = nn.ModuleList()
        self.separate_head_cfg = self.model_cfg.SEPARATE_HEAD_CFG
        for cur_class_names in self.class_names_each_head:
            cur_head_dict = copy.deepcopy(self.separate_head_cfg.HEAD_DICT)
            cur_head_dict['hm'] = dict(out_channels=len(cur_class_names), num_conv=self.model_cfg.NUM_HM_CONV)
            self.heads_list.append(SeparateHead(
                input_channels=self.model_cfg.get('SHARED_CONV_CHANNEL', 128),
                sep_head_dict=cur_head_dict,
                kernel_size=kernel_size_head,
                init_bias=-2.19,
                use_bias=self.model_cfg.get('USE_BIAS_BEFORE_NORM', False),
            ))
        self.predict_boxes_when_training = predict_boxes_when_training
        self.forward_ret_dict = {}
        self.build_losses()

    def build_losses(self):
        self.add_module('hm_loss_func', loss_utils.FocalLossSparse())
        self.add_module('reg_loss_func', loss_utils.RegLossSparse())
        if self.iou_branch:
            self.add_module('crit_iou', IouLossSparse())
            self.add_module('crit_iou_reg', IouRegLossSparse())

    # ------------------------------------------------------------------ targets
    def assign_targets(self, gt_boxes, num_voxels, spatial_indices, spatial_shape):
        """gt_boxes: (B, M, 7 + C + 1), last column the class index (0 = padding)."""
        target_assigner_cfg = self.model_cfg.TARGET_ASSIGNER_CONFIG
        gt_boxes = torch.cat((gt_boxes[:, :, :7], gt_boxes[:, :, -1:]), dim=2)

        batch_size = gt_boxes.shape[0]
        ret_dict = {'heatmaps': [], 'target_boxes': [], 'inds': [], 'masks': [], 'heatmap_masks': [], 'gt_boxes': []}

        all_names = np.array(['bg', *self.class_names])
        for cur_class_names in self.class_names_each_head:
            heatmap_list, target_boxes_list, inds_list, masks_list, gt_boxes_list = [], [], [], [], []
            for bs_idx in range(batch_size):
                cur_gt_boxes = gt_boxes[bs_idx]
                gt_class_names = all_names[cur_gt_boxes[:, -1].cpu().long().numpy()]

                gt_boxes_single_head = []
                for idx, name in enumerate(gt_class_names):
                    if name not in cur_class_names:
                        continue
                    temp_box = cur_gt_boxes[idx].clone()
                    temp_box[-1] = cur_class_names.index(name) + 1
                    gt_boxes_single_head.append(temp_box[None, :])

                if len(gt_boxes_single_head) == 0:
                    gt_boxes_single_head = cur_gt_boxes[:0, :]
                else:
                    gt_boxes_single_head = torch.cat(gt_boxes_single_head, dim=0)

                heatmap, ret_boxes, inds, mask = self.assign_target_of_single_head(
                    num_classes=len(cur_class_names), gt_boxes=gt_boxes_single_head,
                    num_voxels=num_voxels[bs_idx], spatial_indices=spatial_indices[bs_idx],
                    spatial_shape=spatial_shape,
                    feature_map_stride=target_assigner_cfg.FEATURE_MAP_STRIDE,
                    num_max_objs=target_assigner_cfg.NUM_MAX_OBJS,
                    gaussian_overlap=target_assigner_cfg.GAUSSIAN_OVERLAP,
                    min_radius=target_assigner_cfg.MIN_RADIUS,
                )
                heatmap_list.append(heatmap.to(gt_boxes_single_head.device))
                target_boxes_list.append(ret_boxes.to(gt_boxes_single_head.device))
                inds_list.append(inds.to(gt_boxes_single_head.device))
                masks_list.append(mask.to(gt_boxes_single_head.device))
                gt_boxes_list.append(gt_boxes_single_head[:, :-1])

            ret_dict['heatmaps'].append(torch.cat(heatmap_list, dim=1).permute(1, 0))
            ret_dict['target_boxes'].append(torch.stack(target_boxes_list, dim=0))
            ret_dict['inds'].append(torch.stack(inds_list, dim=0))
            ret_dict['masks'].append(torch.stack(masks_list, dim=0))
            ret_dict['gt_boxes'].append(gt_boxes_list)
        return ret_dict

    @staticmethod
    def distance(voxel_indices, center):
        return ((voxel_indices - center.unsqueeze(0)) ** 2).sum(-1)

    def assign_target_of_single_head(self, num_classes, gt_boxes, num_voxels, spatial_indices, spatial_shape,
                                     feature_map_stride, num_max_objs=500, gaussian_overlap=0.1, min_radius=2):
        """
        gt_boxes: (N, 8); spatial_indices: (num_voxels, 2) as (x, y) on the stride-8 grid.
        Each box is assigned to the occupied voxel nearest its centre (VoxelNeXt has no voxel at empty cells).
        """
        heatmap = gt_boxes.new_zeros(num_classes, num_voxels)
        ret_boxes = gt_boxes.new_zeros((num_max_objs, gt_boxes.shape[-1] - 1 + 1))
        inds = gt_boxes.new_zeros(num_max_objs).long()
        mask = gt_boxes.new_zeros(num_max_objs).long()
        if num_voxels == 0:
            return heatmap, ret_boxes, inds, mask

        x, y, z = gt_boxes[:, 0], gt_boxes[:, 1], gt_boxes[:, 2]
        coord_x = (x - self.point_cloud_range[0]) / self.voxel_size[0] / feature_map_stride
        coord_y = (y - self.point_cloud_range[1]) / self.voxel_size[1] / feature_map_stride
        coord_x = torch.clamp(coord_x, min=0, max=spatial_shape[1] - 0.5)
        coord_y = torch.clamp(coord_y, min=0, max=spatial_shape[0] - 0.5)

        center = torch.cat((coord_x[:, None], coord_y[:, None]), dim=-1)
        center_int = center.int()

        dx, dy = gt_boxes[:, 3], gt_boxes[:, 4]
        dx = dx / self.voxel_size[0] / feature_map_stride
        dy = dy / self.voxel_size[1] / feature_map_stride

        radius = centernet_utils.gaussian_radius(dx, dy, min_overlap=gaussian_overlap)
        radius = torch.clamp_min(radius.int(), min=min_radius)

        spatial_indices = spatial_indices.float()
        for k in range(min(num_max_objs, gt_boxes.shape[0])):
            if dx[k] <= 0 or dy[k] <= 0:
                continue
            if not (0 <= center_int[k][0] <= spatial_shape[1] and 0 <= center_int[k][1] <= spatial_shape[0]):
                continue

            cur_class_id = (gt_boxes[k, -1] - 1).long()
            distance = self.distance(spatial_indices, center[k])
            inds[k] = distance.argmin()
            mask[k] = 1

            if 'gt_center' in self.gaussian_type:
                centernet_utils.draw_gaussian_to_heatmap_voxels(
                    heatmap[cur_class_id], distance, radius[k].item() * self.gaussian_ratio)
            if 'nearst' in self.gaussian_type:
                centernet_utils.draw_gaussian_to_heatmap_voxels(
                    heatmap[cur_class_id], self.distance(spatial_indices, spatial_indices[inds[k]]),
                    radius[k].item() * self.gaussian_ratio)

            ret_boxes[k, 0:2] = center[k] - spatial_indices[inds[k]][:2]
            ret_boxes[k, 2] = z[k]
            ret_boxes[k, 3:6] = gt_boxes[k, 3:6].log()
            ret_boxes[k, 6] = torch.cos(gt_boxes[k, 6])
            ret_boxes[k, 7] = torch.sin(gt_boxes[k, 6])
            if gt_boxes.shape[1] > 8:
                ret_boxes[k, 8:] = gt_boxes[k, 7:-1]
        return heatmap, ret_boxes, inds, mask

    # ------------------------------------------------------------------ loss
    @staticmethod
    def sigmoid(x):
        return torch.clamp(x.sigmoid(), min=1e-4, max=1 - 1e-4)

    def get_loss(self):
        pred_dicts = self.forward_ret_dict['pred_dicts']
        target_dicts = self.forward_ret_dict['target_dicts']
        batch_index = self.forward_ret_dict['batch_index']

        tb_dict = {}
        loss = 0
        batch_indices = self.forward_ret_dict['voxel_indices'][:, 0]
        spatial_indices = self.forward_ret_dict['voxel_indices'][:, 1:]
        loss_weights = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS

        for idx, pred_dict in enumerate(pred_dicts):
            pred_dict['hm'] = self.sigmoid(pred_dict['hm'])
            hm_loss = self.hm_loss_func(pred_dict['hm'], target_dicts['heatmaps'][idx]) * loss_weights['cls_weight']

            target_boxes = target_dicts['target_boxes'][idx]
            pred_boxes = torch.cat([pred_dict[head_name] for head_name in self.separate_head_cfg.HEAD_ORDER], dim=1)

            reg_loss = self.reg_loss_func(
                pred_boxes, target_dicts['masks'][idx], target_dicts['inds'][idx], target_boxes, batch_index
            )
            loc_loss = (reg_loss * reg_loss.new_tensor(loss_weights['code_weights'])).sum()
            loc_loss = loc_loss * loss_weights['loc_weight']
            tb_dict['hm_loss_head_%d' % idx] = hm_loss.item()
            tb_dict['loc_loss_head_%d' % idx] = loc_loss.item()
            if self.iou_branch:
                batch_box_preds = self._get_predicted_boxes(pred_dict, spatial_indices)
                iou_loss = self.crit_iou(pred_dict['iou'], target_dicts['masks'][idx], target_dicts['inds'][idx],
                                         batch_box_preds.detach(), target_dicts['gt_boxes'][idx], batch_indices)
                iou_reg_loss = self.crit_iou_reg(batch_box_preds, target_dicts['masks'][idx],
                                                 target_dicts['inds'][idx], target_dicts['gt_boxes'][idx],
                                                 batch_indices)
                iou_weight = loss_weights['iou_weight'] if 'iou_weight' in loss_weights else loss_weights['loc_weight']
                iou_reg_loss = iou_reg_loss * iou_weight
                loss = loss + hm_loss + loc_loss + iou_loss + iou_reg_loss
                tb_dict['iou_loss_head_%d' % idx] = iou_loss.item()
                tb_dict['iou_reg_loss_head_%d' % idx] = iou_reg_loss.item()
            else:
                loss = loss + hm_loss + loc_loss

        tb_dict['rpn_loss'] = loss.item()
        return loss, tb_dict

    def _get_predicted_boxes(self, pred_dict, spatial_indices):
        center = pred_dict['center']
        center_z = pred_dict['center_z']
        dim = torch.exp(torch.clamp(pred_dict['dim'], min=-5, max=5))
        rot_cos = pred_dict['rot'][:, 0].unsqueeze(dim=1)
        rot_sin = pred_dict['rot'][:, 1].unsqueeze(dim=1)
        angle = torch.atan2(rot_sin, rot_cos)
        xs = (spatial_indices[:, 1:2] + center[:, 0:1]) * self.feature_map_stride * self.voxel_size[0] \
            + self.point_cloud_range[0]
        ys = (spatial_indices[:, 0:1] + center[:, 1:2]) * self.feature_map_stride * self.voxel_size[1] \
            + self.point_cloud_range[1]
        return torch.cat([xs, ys, center_z, dim, angle], dim=-1)

    # ------------------------------------------------------------------ inference
    def rotate_class_specific_nms_iou(self, boxes, scores, iou_preds, labels, rectifier, nms_configs):
        box_preds_list, scores_list, labels_list = [], [], []
        for cls in range(self.num_class):
            mask = labels == cls
            boxes_cls = boxes[mask]
            scores_cls = torch.pow(scores[mask], 1 - rectifier[cls]) * \
                torch.pow(iou_preds[mask].squeeze(-1), rectifier[cls])
            labels_cls = labels[mask]

            selected, _ = model_nms_utils.class_agnostic_nms(
                box_scores=scores_cls, box_preds=boxes_cls, nms_config=nms_configs[cls], score_thresh=None)

            box_preds_list.append(boxes_cls[selected])
            scores_list.append(scores_cls[selected])
            labels_list.append(labels_cls[selected])
        return torch.cat(box_preds_list, dim=0), torch.cat(scores_list, dim=0), torch.cat(labels_list, dim=0)

    def generate_predicted_boxes(self, batch_size, pred_dicts, voxel_indices, spatial_shape):
        post_process_cfg = self.model_cfg.POST_PROCESSING
        post_center_limit_range = torch.tensor(post_process_cfg.POST_CENTER_LIMIT_RANGE).cuda().float()

        ret_dict = [{'pred_boxes': [], 'pred_scores': [], 'pred_labels': [], 'pred_ious': []}
                    for _ in range(batch_size)]
        for idx, pred_dict in enumerate(pred_dicts):
            batch_hm = pred_dict['hm'].sigmoid()
            batch_center = pred_dict['center']
            batch_center_z = pred_dict['center_z']
            batch_dim = pred_dict['dim'].exp()
            batch_rot_cos = pred_dict['rot'][:, 0].unsqueeze(dim=1)
            batch_rot_sin = pred_dict['rot'][:, 1].unsqueeze(dim=1)
            # IoU branch output is in [-1, 1]; clamp so pow() below never sees a negative base
            batch_iou = ((pred_dict['iou'] + 1) * 0.5).clamp(min=0, max=1) if self.iou_branch else None
            batch_vel = pred_dict['vel'] if 'vel' in self.separate_head_cfg.HEAD_ORDER else None

            final_pred_dicts = centernet_utils.decode_bbox_from_voxels_nuscenes(
                batch_size=batch_size, indices=voxel_indices,
                obj=batch_hm,
                rot_cos=batch_rot_cos, rot_sin=batch_rot_sin,
                center=batch_center, center_z=batch_center_z,
                dim=batch_dim, vel=batch_vel, iou=batch_iou,
                point_cloud_range=self.point_cloud_range, voxel_size=self.voxel_size,
                feature_map_stride=self.feature_map_stride,
                K=post_process_cfg.MAX_OBJ_PER_SAMPLE,
                score_thresh=post_process_cfg.SCORE_THRESH,
                post_center_limit_range=post_center_limit_range,
            )

            for k, final_dict in enumerate(final_pred_dicts):
                final_dict['pred_labels'] = self.class_id_mapping_each_head[idx][final_dict['pred_labels'].long()]
                if not self.iou_branch:
                    selected, selected_scores = model_nms_utils.class_agnostic_nms(
                        box_scores=final_dict['pred_scores'], box_preds=final_dict['pred_boxes'],
                        nms_config=post_process_cfg.NMS_CONFIG, score_thresh=None)
                    final_dict['pred_boxes'] = final_dict['pred_boxes'][selected]
                    final_dict['pred_scores'] = selected_scores
                    final_dict['pred_labels'] = final_dict['pred_labels'][selected]

                ret_dict[k]['pred_boxes'].append(final_dict['pred_boxes'])
                ret_dict[k]['pred_scores'].append(final_dict['pred_scores'])
                ret_dict[k]['pred_labels'].append(final_dict['pred_labels'])
                ret_dict[k]['pred_ious'].append(final_dict['pred_ious'])

        for k in range(batch_size):
            pred_boxes = torch.cat(ret_dict[k]['pred_boxes'], dim=0)
            pred_scores = torch.cat(ret_dict[k]['pred_scores'], dim=0)
            pred_labels = torch.cat(ret_dict[k]['pred_labels'], dim=0)
            if self.iou_branch:
                pred_ious = torch.cat(ret_dict[k]['pred_ious'], dim=0)
                pred_boxes, pred_scores, pred_labels = self.rotate_class_specific_nms_iou(
                    pred_boxes, pred_scores, pred_ious, pred_labels, self.rectifier, self.nms_configs)
            ret_dict[k]['pred_boxes'] = pred_boxes
            ret_dict[k]['pred_scores'] = pred_scores
            ret_dict[k]['pred_labels'] = pred_labels + 1
        return ret_dict

    @staticmethod
    def _get_voxel_infos(x):
        spatial_shape = x.spatial_shape
        voxel_indices = x.indices
        batch_index = voxel_indices[:, 0]
        spatial_indices, num_voxels = [], []
        for bs_idx in range(x.batch_size):
            batch_inds = batch_index == bs_idx
            spatial_indices.append(voxel_indices[batch_inds][:, [2, 1]])
            num_voxels.append(int(batch_inds.sum()))
        return spatial_shape, batch_index, voxel_indices, spatial_indices, num_voxels

    def forward(self, data_dict):
        x = data_dict['encoded_spconv_tensor']

        spatial_shape, batch_index, voxel_indices, spatial_indices, num_voxels = self._get_voxel_infos(x)
        self.forward_ret_dict['batch_index'] = batch_index

        pred_dicts = [head(x) for head in self.heads_list]

        if self.training:
            self.forward_ret_dict['target_dicts'] = self.assign_targets(
                data_dict['gt_boxes'], num_voxels, spatial_indices, spatial_shape)

        self.forward_ret_dict['pred_dicts'] = pred_dicts
        self.forward_ret_dict['voxel_indices'] = voxel_indices

        if not self.training or self.predict_boxes_when_training:
            data_dict['final_box_dicts'] = self.generate_predicted_boxes(
                data_dict['batch_size'], pred_dicts, voxel_indices, spatial_shape)
        return data_dict
