"""
Shared helpers for the analysis scripts (feature extraction, Grad-CAM): load a config + checkpoint, pick validation
frames, run the model, and read out the encoder's bird's-eye-view (BEV) feature map.

The scripts change into detection/tools, because the configs refer to each other by paths relative to it.
"""
import os
import sys

import numpy as np
import torch

TOOLS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
REPO_ROOT = os.path.abspath(os.path.join(TOOLS_DIR, '..', '..'))
for p in (os.path.join(REPO_ROOT, 'utils'), os.path.join(REPO_ROOT, 'detection')):
    if p not in sys.path:
        sys.path.insert(0, p)

from al3d_utils import common_utils  # noqa: E402
from al3d_utils.config_utils import cfg, cfg_from_list, cfg_from_yaml_file  # noqa: E402
from al3d_utils.model_utils import load_params_from_file  # noqa: E402
from al3d_utils.ops.iou3d_nms import iou3d_nms_utils  # noqa: E402
from al3d_det.datasets import __all__ as dataset_classes  # noqa: E402
from al3d_det.models import build_network, load_data_to_gpu  # noqa: E402

DEFAULT_CFG = 'cfgs/det_model_cfgs/dsert/lidar_only.yaml'
# IoU needed for a ground-truth box to count as detected (the DSERT/Waymo thresholds used by the evaluation).
MATCH_IOU = {'Vehicle': 0.5, 'Pedestrian': 0.3, 'Bike': 0.3}


def add_common_args(parser):
    parser.add_argument('--ckpt', required=True, help='checkpoint .pth (full or weights-only)')
    parser.add_argument('--sensor', required=True, choices=['livox', 'ouster', 'radar'],
                        help='which point cloud to feed the model')
    parser.add_argument('--data_root', required=True, help='DSERT-RoLL root (contains ImageSets/ and processed_data/)')
    parser.add_argument('--cfg_file', default=DEFAULT_CFG)
    parser.add_argument('--split', default='test', choices=['train', 'test'], help="'test' = the val split")
    parser.add_argument('--max_frames', type=int, default=400, help='frames sampled evenly across the split')
    parser.add_argument('--weather', default=None, help='only frames of this weather (e.g. Fog)')
    return parser


def setup(args):
    """Load config, dataset and model. Returns (cfg, dataset, model, logger)."""
    args.ckpt = os.path.abspath(args.ckpt)
    args.data_root = os.path.abspath(args.data_root)
    os.chdir(TOOLS_DIR)
    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg_from_list(['DATA_CONFIG.DATA_PATH', args.data_root, 'DATA_CONFIG.SENSOR.NAME', args.sensor], cfg)
    logger = common_utils.create_logger()

    dataset = dataset_classes[cfg.DATA_CONFIG.DATASET](
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, root_path=None,
        training=(args.split == 'train'), logger=logger)
    model = build_network(cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    load_params_from_file(model, filename=args.ckpt, logger=logger, to_cpu=False)
    model.cuda().eval()
    if hasattr(model, 'set_amp'):
        model.set_amp(None)  # analysis in full fp32: gradients for Grad-CAM, stable features for t-SNE
    return cfg, dataset, model, logger


def select_frames(dataset, max_frames, weather=None):
    idx = [i for i, info in enumerate(dataset.infos) if weather is None or info.get('weather') == weather]
    if len(idx) > max_frames:
        idx = [idx[int(k)] for k in np.linspace(0, len(idx) - 1, max_frames)]
    return idx


def load_frame(dataset, i):
    batch = dataset.collate_batch([dataset[i]])
    load_data_to_gpu(batch)
    return batch


def gt_in_range(batch, pc_range):
    """Ground-truth boxes (M, 8) [x, y, z, dx, dy, dz, heading, class] of sample 0 that lie inside the point cloud range."""
    gt = batch['gt_boxes'][0]
    gt = gt[gt[:, 3:6].sum(1) > 0]  # drop zero padding
    keep = (gt[:, 0] >= pc_range[0]) & (gt[:, 0] < pc_range[3]) & (gt[:, 1] >= pc_range[1]) & (gt[:, 1] < pc_range[4])
    return gt[keep]


def points_in_boxes(points, boxes):
    """points (N, 3), boxes (M, 7) -> (M,) number of points inside each box."""
    if len(boxes) == 0 or len(points) == 0:
        return torch.zeros(len(boxes), dtype=torch.long, device=boxes.device)
    local = points[None, :, :3] - boxes[:, None, :3]                    # (M, N, 3)
    c, s = torch.cos(-boxes[:, 6])[:, None], torch.sin(-boxes[:, 6])[:, None]
    lx = local[..., 0] * c - local[..., 1] * s
    ly = local[..., 0] * s + local[..., 1] * c
    half = boxes[:, None, 3:6] / 2
    inside = (lx.abs() <= half[..., 0]) & (ly.abs() <= half[..., 1]) & (local[..., 2].abs() <= half[..., 2])
    return inside.sum(1)


def sample_bev(feat, xy, pc_range):
    """Bilinearly sample a BEV feature map feat (C, H, W) at metric positions xy (M, 2) -> (M, C).
    Rows of the map run along y and columns along x, covering the point cloud range."""
    gx = (xy[:, 0] - pc_range[0]) / (pc_range[3] - pc_range[0]) * 2 - 1
    gy = (xy[:, 1] - pc_range[1]) / (pc_range[4] - pc_range[1]) * 2 - 1
    grid = torch.stack([gx, gy], -1).view(1, 1, -1, 2).to(feat.dtype)
    out = torch.nn.functional.grid_sample(feat[None], grid, mode='bilinear', align_corners=False)
    return out[0, :, 0].t()


def match_detections(pred, gt, class_names):
    """For each GT box, the best 3D IoU with a same-class prediction, its score, and whether it counts as detected."""
    m = len(gt)
    best_iou = torch.zeros(m, device=gt.device)
    best_score = torch.zeros(m, device=gt.device)
    if m and len(pred['pred_boxes']):
        iou = iou3d_nms_utils.boxes_iou3d_gpu(pred['pred_boxes'][:, :7].float(), gt[:, :7].float())   # (P, M)
        same = pred['pred_labels'].long()[:, None] == gt[:, 7].long()[None, :]
        iou = torch.where(same, iou, torch.zeros_like(iou))
        best_iou, arg = iou.max(0)
        best_score = pred['pred_scores'][arg] * (best_iou > 0)
    thr = torch.tensor([MATCH_IOU.get(class_names[int(c) - 1], 0.5) for c in gt[:, 7]], device=gt.device)
    return best_iou, best_score, best_iou >= thr


def run_model(model, batch):
    """Full inference (eval mode). Returns predictions of sample 0; batch keeps the intermediate tensors,
    e.g. batch['spatial_features_2d'] (1, C, H, W): the encoder output at 1/8 resolution."""
    with torch.no_grad():
        pred_dicts, _ = model(batch)
    return pred_dicts[0]
