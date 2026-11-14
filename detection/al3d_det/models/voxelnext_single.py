import torch

from al3d_det.models.modules import backbone_3d, dense_heads
from .centerpoint_waymo import CenterPointPC
from .centerpoint_single import _cast_to_fp32


class VoxelNeXtSingle(CenterPointPC):
    """
    VoxelNeXt (Chen et al., CVPR 2023) on one point-cloud sensor: VFE -> VoxelResBackBone8xVoxelNeXt -> VoxelNeXtHead.
    Fully sparse and single-stage: no dense BEV map, no 2D backbone, no ROI head.

    Same interface as CenterPointSingle (set_amp, compile_dense, encoder = vfe + backbone3d), so
    train_single_sensor.sbatch, eval_ckpt.sbatch, finetune_from_best.sbatch and --pretrained_model work unchanged
    with CFG=cfgs/det_model_cfgs/dsert/voxelnext.yaml. Which sensor feeds it is DATA_CONFIG.SENSOR, so a Livox
    checkpoint loads into a radar run (same parameter names and shapes, 4 input channels in both).
    """
    ENCODER_MODULES = ('vfe', 'backbone3d')

    def __init__(self, model_cfg, num_class, dataset):
        assert not model_cfg.get('SECOND_STAGE', False), 'VoxelNeXt is single-stage: set SECOND_STAGE False'
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
        dense_head = dense_heads.__all__[cfg.DENSE_HEAD.NAME](
            model_cfg=cfg.DENSE_HEAD,
            input_channels=backbone3d.num_point_features,
            num_class=self.num_class if not cfg.DENSE_HEAD.CLASS_AGNOSTIC else 1,
            class_names=self.class_names,
            grid_size=grid_size,
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            predict_boxes_when_training=False,
        )
        module_list = []
        for name, module in [('vfe', vfe), ('backbone3d', backbone3d), ('dense_head', dense_head)]:
            self.add_module(name, module)
            module_list.append(module)
        return module_list

    def set_amp(self, dtype):
        """dtype: torch.float16, torch.bfloat16, or None to run everything in fp32."""
        self.amp_dtype = dtype

    def compile_dense(self, **compile_kwargs):
        """Every layer is a spconv kernel, which dynamo cannot trace: nothing to compile (--compile is a no-op)."""
        return 0

    def forward(self, batch_dict):
        # Mixed precision covers the sparse backbone during training only; the head (target assignment, decoding,
        # NMS) always runs in fp32. Evaluation runs entirely in fp32: in eval mode spconv skips the autocast-aware
        # autograd path, and the 5x5x5 strided convolutions then fail under autocast ("can't find suitable algorithm",
        # seen on H100 with spconv 2.3). This also makes evaluation independent of the --amp setting used in training.
        use_amp = self.training and self.amp_dtype is not None
        with torch.autocast('cuda', dtype=self.amp_dtype or torch.float16, enabled=use_amp):
            for name in self.ENCODER_MODULES:
                batch_dict = getattr(self, name)(batch_dict)
        if use_amp:
            _cast_to_fp32(batch_dict)

        batch_dict = self.dense_head(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()
            return {'loss': loss}, tb_dict, disp_dict
        pred_dicts, recall_dicts = self.post_processing(batch_dict)
        return pred_dicts, recall_dicts
