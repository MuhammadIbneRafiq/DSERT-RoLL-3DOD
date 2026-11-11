"""
What does the network's 4th input channel actually contain, per sensor? CPU only, a few minutes on a login node.

Goes through DSERTSingleSensorDataset (the exact loader training uses) and reports, for each sensor:
  * the raw fields stored in the .npy files and which one became the 4th channel
  * the 4th channel per point and per voxel (DynamicMeanVFE feeds the per-voxel mean of x, y, z, feature into the
    first 3D conv, unnormalised), next to x / y / z for scale
  * for every other stored field (e.g. radar RCS), the same summary, to see whether a better intensity analogue exists

    cd detection/tools
    python analysis/channel_stats.py --data_root $HOME/data/dsert_roll --sensors livox radar --num_frames 200
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from al3d_utils import common_utils  # noqa: E402
from al3d_utils.config_utils import cfg, cfg_from_list, cfg_from_yaml_file  # noqa: E402
from al3d_det.datasets import __all__ as dataset_classes  # noqa: E402


def summary(x):
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return 'empty'
    p = np.percentile(x, [1, 5, 50, 95, 99])
    return ('n=%-9d min %9.3f  p1 %9.3f  p5 %9.3f  p50 %9.3f  p95 %9.3f  p99 %9.3f  max %9.3f  | mean %8.3f  std %8.3f'
            '  | zero %5.1f%%  neg %5.1f%%' % (x.size, x.min(), *p, x.max(), x.mean(), x.std(),
                                             100 * np.mean(x == 0), 100 * np.mean(x < 0)))


def voxel_means(pts, pc_range, voxel_size):
    """Per-voxel mean of (x, y, z, feature), as DynamicMeanVFE computes it. Also returns points per voxel."""
    lo, hi = np.asarray(pc_range[:3]), np.asarray(pc_range[3:])
    pts = pts[np.all((pts[:, :3] >= lo) & (pts[:, :3] < hi), 1)]
    if len(pts) == 0:
        return np.zeros((0, 4)), np.zeros(0)
    idx = np.floor((pts[:, :3] - lo) / voxel_size).astype(np.int64)
    _, inv, cnt = np.unique(idx, axis=0, return_inverse=True, return_counts=True)
    inv = inv.reshape(-1)
    sums = np.zeros((len(cnt), 4))
    np.add.at(sums, inv, pts[:, :4])
    return sums / cnt[:, None], cnt


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data_root', required=True)
    ap.add_argument('--cfg_file', default='cfgs/det_model_cfgs/dsert/lidar_only.yaml')
    ap.add_argument('--sensors', nargs='+', default=['livox', 'radar'])
    ap.add_argument('--num_frames', type=int, default=200)
    args = ap.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg_from_list(['DATA_CONFIG.DATA_PATH', args.data_root], cfg)
    logger = common_utils.create_logger()
    voxel_size = np.asarray([p for p in cfg.DATA_CONFIG.DATA_PROCESSOR
                             if p.NAME.startswith('transform_points_to_voxels')][0].VOXEL_SIZE)

    for sensor in args.sensors:
        cfg.DATA_CONFIG.SENSOR.NAME = sensor
        ds = dataset_classes[cfg.DATA_CONFIG.DATASET](dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES,
                                                      root_path=None, training=True, logger=logger)
        rng = np.random.default_rng(0)
        idxs = sorted(rng.choice(len(ds.infos), size=min(len(ds.infos), args.num_frames), replace=False).tolist())

        pts_all, vox_all, cnt_all, raw_fields, n_empty = [], [], [], {}, 0
        names = None
        for i in idxs:
            _, pts_list, _ = ds.get_infos_and_points([i])
            pts = pts_list[0][:-1]                       # drop the sentinel point
            n_empty += len(pts) == 0
            pts_all.append(pts)
            v, c = voxel_means(pts, ds.point_cloud_range, voxel_size)
            vox_all.append(v)
            cnt_all.append(c)
            raw = np.load(os.path.join(ds.data_path, ds.sensor_rel_path(ds.infos[i])))
            if raw.size and raw.dtype.names:
                names = raw.dtype.names
                for f in names:
                    if f not in ('x', 'y', 'z') and np.issubdtype(raw.dtype[f], np.number):
                        raw_fields.setdefault(f, []).append(raw[f].reshape(-1))

        P, V, C = np.concatenate(pts_all), np.concatenate(vox_all), np.concatenate(cnt_all)
        print('\n' + '=' * 30 + ' %s: %d frames (%d empty) ' % (sensor, len(idxs), n_empty) + '=' * 30)
        print('stored fields: %s | 4th channel = "%s" | transform = %s'
              % (names, ds._feature_name, ds.sensor_opts['INTENSITY_TRANSFORM']))
        print('points/frame median %d | voxels/frame median %d | points per voxel: median %.0f, mean %.2f'
              % (np.median([len(p) for p in pts_all]), np.median([len(v) for v in vox_all]), np.median(C), C.mean()))
        print('-- per point (Livox frame, after transform) --')
        for k, name in enumerate(['x', 'y', 'z', '4th']):
            print('  %-4s %s' % (name, summary(P[:, k])))
        print('-- per voxel mean = what the first conv sees --')
        for k, name in enumerate(['x', 'y', 'z', '4th']):
            print('  %-4s %s' % (name, summary(V[:, k])))
        others = {f: v for f, v in raw_fields.items() if f != ds._feature_name}
        if others:
            print('-- other stored fields (raw, not fed to the model) --')
            for f, v in others.items():
                print('  %-12s %s' % (f, summary(np.concatenate(v))))


if __name__ == '__main__':
    main()
