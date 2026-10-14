"""
GPU self-test of the single-sensor model on random points, no dataset needed. Runs a few training steps
(mixed precision + torch.compile, as in real training) and one inference pass, so compiled CUDA ops, spconv,
AMP and compile are all exercised on the GPU before a real job is submitted.

    python gpu_selftest.py --cfg_file cfgs/det_model_cfgs/dsert/lidar_only.yaml --amp fp16 --compile
"""
import argparse
import time

import numpy as np
import torch

from al3d_utils.config_utils import cfg, cfg_from_yaml_file
from al3d_det.datasets.processor.data_processor import DataProcessor
from al3d_det.datasets.processor.point_feature_encoder import PointFeatureEncoder
from al3d_det.models import build_network


class GridOnlyDataset:
    """The attributes build_network reads from a dataset, taken from the config."""

    def __init__(self, data_cfg, class_names):
        self.class_names = class_names
        self.point_cloud_range = np.array(data_cfg.POINT_CLOUD_RANGE, dtype=np.float32)
        self.point_feature_encoder = PointFeatureEncoder(data_cfg.POINT_FEATURE_ENCODING, self.point_cloud_range)
        processor = DataProcessor(data_cfg.DATA_PROCESSOR, self.point_cloud_range, training=True,
                                  num_point_features=self.point_feature_encoder.num_point_features)
        self.grid_size, self.voxel_size = processor.grid_size, processor.voxel_size


def random_batch(pc_range, batch_size, num_classes, device, rng):
    """Background points in range, plus dense point clusters inside a few labelled boxes per sample."""
    points, boxes = [], []
    for b in range(batch_size):
        n_bg = 30000
        bg = np.c_[rng.uniform(pc_range[0], pc_range[3], n_bg), rng.uniform(pc_range[1], pc_range[4], n_bg),
                   rng.uniform(pc_range[2], pc_range[5], n_bg)]
        sample_boxes = []
        for k in range(6):
            cls = k % num_classes + 1
            dims = [[4.2, 1.9, 1.6], [0.8, 0.8, 1.8], [1.8, 0.7, 1.6]][cls - 1]
            box = [rng.uniform(5, 60), rng.uniform(-30, 30), 0.0, *dims, rng.uniform(-np.pi, np.pi), cls]
            sample_boxes.append(box)
            c, s = np.cos(box[6]), np.sin(box[6])
            local = (rng.random((400, 3)) - 0.5) * np.array(dims)
            bg = np.r_[bg, np.c_[local[:, 0] * c - local[:, 1] * s + box[0], local[:, 0] * s + local[:, 1] * c + box[1],
                                 local[:, 2] + box[2]]]
        feat = rng.uniform(0, 8, (len(bg), 1))
        points.append(np.c_[np.full((len(bg), 1), b), bg, feat])
        boxes.append(sample_boxes)
    return {
        'points': torch.tensor(np.concatenate(points), dtype=torch.float32, device=device),
        'gt_boxes': torch.tensor(np.array(boxes), dtype=torch.float32, device=device),
        'batch_size': batch_size,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--cfg_file', default='cfgs/det_model_cfgs/dsert/lidar_only.yaml')
    parser.add_argument('--amp', choices=['none', 'fp16', 'bf16'], default='fp16')
    parser.add_argument('--compile', action='store_true')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--steps', type=int, default=6)
    args = parser.parse_args()

    assert torch.cuda.is_available(), 'no GPU visible'
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    cfg_from_yaml_file(args.cfg_file, cfg)
    device = torch.device('cuda')
    print('GPU:', torch.cuda.get_device_name(0), '| torch', torch.__version__, '| cuda', torch.version.cuda)

    dataset = GridOnlyDataset(cfg.DATA_CONFIG, cfg.CLASS_NAMES)
    model = build_network(cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset).cuda()
    print('model: %s, %.1fM parameters, grid %s' % (cfg.MODEL.NAME, sum(p.numel() for p in model.parameters()) / 1e6,
                                                   dataset.grid_size.tolist()))
    amp_dtype = {'none': None, 'fp16': torch.float16, 'bf16': torch.bfloat16}[args.amp]
    model.set_amp(amp_dtype)
    if args.compile:
        print('compiled %d dense submodules' % model.compile_dense())
    scaler = torch.cuda.amp.GradScaler() if args.amp == 'fp16' else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    rng = np.random.default_rng(0)

    model.train()
    times = []
    for step in range(args.steps):
        batch = random_batch(dataset.point_cloud_range, args.batch_size, len(cfg.CLASS_NAMES), device, rng)
        torch.cuda.synchronize()
        t0 = time.time()
        ret, tb_dict, _ = model(batch)
        loss = ret['loss'].mean()
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()
        torch.cuda.synchronize()
        times.append(time.time() - t0)
        assert torch.isfinite(loss), 'non-finite loss at step %d: %s' % (step, tb_dict)
        print('train step %d: loss %.4f  (%.2fs%s)' % (step, loss.item(), times[-1], ', includes compile' if step == 0 else ''))

    model.eval()
    with torch.no_grad():
        pred_dicts, _ = model(random_batch(dataset.point_cloud_range, args.batch_size, len(cfg.CLASS_NAMES), device, rng))
    assert len(pred_dicts) == args.batch_size and {'pred_boxes', 'pred_scores', 'pred_labels'} <= set(pred_dicts[0])
    print('inference: %d samples, %s boxes in sample 0' % (len(pred_dicts), tuple(pred_dicts[0]['pred_boxes'].shape)))

    steady = times[2:] or times
    print('steady-state: %.2f s/step at batch %d (synthetic data, rough guide only); peak GPU memory %.1f GiB'
          % (float(np.mean(steady)), args.batch_size, torch.cuda.max_memory_allocated() / 2 ** 30))
    print('GPU SELF-TEST PASSED')


if __name__ == '__main__':
    main()
