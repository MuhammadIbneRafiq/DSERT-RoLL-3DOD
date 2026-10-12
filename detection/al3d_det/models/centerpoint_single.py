import torch

from al3d_det.models.modules import backbone_3d, backbone_2d, dense_heads, roi_heads
from .centerpoint_waymo import CenterPointPC

try:
    import spconv.pytorch as spconv
except ImportError:
    spconv = None


class CenterPointSingle(CenterPointPC):
    """
    The LiDAR path of CenterPointMM on its own: VFE -> sparse 3D backbone -> BEV -> 2D backbone ->
    CenterHead -> LoGoHead. No radar branch and no cameras.

    Which sensor feeds it is a dataset choice (DATA_CONFIG.SENSOR), and parameter names match the
    `lidar.*` submodule of CenterPointMM, so the same checkpoint loads into a LiDAR run, a radar run,
    or the LiDAR branch of the fusion model.
    """
    ENCODER_MODULES = ('vfe', 'backbone3d', 'map_to_bev', 'backbone2d')

    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg, num_class, dataset)
        self.amp_dtype = None

    def build_networks(self):
        cfg = self.model_cfg
        grid_size = self.dataset.grid_size
        voxel_size = self.dataset.voxel_size
        point_cloud_range = self.dataset.point_cloud_range

        vfe = backbone_3d.__all__[cfg.VFE.NAME](
            model_cfg=cfg.VFE,
            num_point_features=self.dataset.point_feature_encoder.num_point_features,
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            grid_size=grid_size,
        )
        backbone3d = backbone_3d.__all__[cfg.BACKBONE_3D.NAME](
            model_cfg=cfg.BACKBONE_3D,
            input_channels=vfe.get_output_feature_dim(),
            grid_size=grid_size,
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
        )
        map_to_bev = backbone_2d.__all__[cfg.MAP_TO_BEV.NAME](
            model_cfg=cfg.MAP_TO_BEV,
            grid_size=grid_size
        )
        backbone2d = backbone_2d.__all__[cfg.BACKBONE_2D.NAME](
            model_cfg=cfg.BACKBONE_2D,
            input_channels=map_to_bev.num_bev_features
        )
        dense_head = dense_heads.__all__[cfg.DENSE_HEAD.NAME](
            model_cfg=cfg.DENSE_HEAD,
            input_channels=backbone2d.num_bev_features,
            num_class=self.num_class if not cfg.DENSE_HEAD.CLASS_AGNOSTIC else 1,
            class_names=self.class_names,
            grid_size=grid_size,
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            predict_boxes_when_training=self.second_stage
        )

        module_list = []
        for name, module in [('vfe', vfe), ('backbone3d', backbone3d), ('map_to_bev', map_to_bev),
                             ('backbone2d', backbone2d), ('dense_head', dense_head)]:
            self.add_module(name, module)
            module_list.append(module)

        if self.second_stage:
            roi_head = roi_heads.__all__[cfg.ROI_HEAD.NAME](
                model_cfg=cfg.ROI_HEAD,
                input_channels=backbone3d.num_point_features,
                num_class=self.num_class if not cfg.ROI_HEAD.CLASS_AGNOSTIC else 1,
                grid_size=grid_size,
                voxel_size=voxel_size,
                point_cloud_range=point_cloud_range
            )
            self.add_module('roi_head', roi_head)
            module_list.append(roi_head)

        return module_list

    def set_amp(self, dtype):
        """dtype: torch.float16, torch.bfloat16, or None to run everything in fp32."""
        self.amp_dtype = dtype

    def compile_dense(self, **compile_kwargs):
        """
        torch.compile the dense, tensor-in/tensor-out parts in place (parameter names do not change).
        The sparse 3D backbone (spconv) and the ROI pooling/NMS CUDA ops are custom kernels that
        dynamo cannot trace, so they stay eager.
        """
        targets = list(self.backbone2d.blocks) + list(self.backbone2d.deblocks)
        targets += [self.dense_head.shared_conv] + list(self.dense_head.heads_list)
        if self.second_stage:
            targets += [self.roi_head.shared_fc_layer, self.roi_head.cls_layers, self.roi_head.reg_layers]
        for module in targets:
            module.compile(**compile_kwargs)
        return len(targets)

    def forward(self, batch_dict):
        # Mixed precision covers the encoder, where the compute is. The heads run in fp32 (TF32 on
        # Ampere/Hopper) because target assignment, NMS and the pointnet2 pooling kernels expect float32.
        with torch.autocast('cuda', dtype=self.amp_dtype or torch.float16, enabled=self.amp_dtype is not None):
            for name in self.ENCODER_MODULES:
                batch_dict = getattr(self, name)(batch_dict)
        if self.amp_dtype is not None:
            _cast_to_fp32(batch_dict)

        batch_dict = self.dense_head(batch_dict)
        if self.second_stage:
            batch_dict = self.roi_head(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()
            return {'loss': loss}, tb_dict, disp_dict
        pred_dicts, recall_dicts = self.post_processing(batch_dict)
        return pred_dicts, recall_dicts


def _cast_to_fp32(obj):
    """Cast half-precision tensors (including spconv feature tensors) inside a batch_dict to fp32, in place."""
    items = obj.items() if isinstance(obj, dict) else enumerate(obj)
    for key, val in list(items):
        if torch.is_tensor(val):
            if val.is_floating_point() and val.dtype != torch.float32:
                obj[key] = val.float()
        elif spconv is not None and isinstance(val, spconv.SparseConvTensor):
            if val.features.dtype != torch.float32:
                obj[key] = val.replace_feature(val.features.float())
        elif isinstance(val, (dict, list)):
            _cast_to_fp32(val)
