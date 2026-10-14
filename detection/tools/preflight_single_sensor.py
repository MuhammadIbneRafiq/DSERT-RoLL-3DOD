"""
Sanity-check a single-sensor DSERT config before spending GPU hours on it. CPU only.

For a sample of frames it reports:
  * which file key / calibration entry the dataset resolved for the sensor, and the sensor -> Livox transform
  * point counts and the 4th-channel (intensity) distribution after the configured transform
  * how many labelled boxes contain points, for this sensor and for the Livox (the sensor the labels were
    made with), by distance. If the sensor is far below the Livox at short range, the extrinsic is wrong.
  * optionally a bird's-eye-view PNG per frame (needs matplotlib)

Exits non-zero on hard failures (missing files, no points, non-finite values). Coverage problems are
reported as warnings, because a short-range sensor legitimately sees fewer distant boxes.

Example:
    python preflight_single_sensor.py --cfg_file cfgs/det_model_cfgs/dsert/lidar_only.yaml --num_frames 20 --plot_dir preflight_png
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

from al3d_utils import common_utils
from al3d_utils.config_utils import cfg, cfg_from_list, cfg_from_yaml_file
from al3d_det.datasets import __all__ as dataset_classes

DIST_BINS = [0, 15, 30, 50, 80]


def points_in_boxes_count(points, boxes):
    """points (N, 3+), boxes (M, 7) [x, y, z, dx, dy, dz, heading] -> (M,) number of points inside each box."""
    counts = np.zeros(len(boxes), dtype=np.int64)
    for i, (x, y, z, dx, dy, dz, h) in enumerate(boxes[:, :7]):
        local = points[:, :3] - np.array([x, y, z])
        c, s = np.cos(-h), np.sin(-h)
        lx = local[:, 0] * c - local[:, 1] * s
        ly = local[:, 0] * s + local[:, 1] * c
        inside = (np.abs(lx) <= dx / 2) & (np.abs(ly) <= dy / 2) & (np.abs(local[:, 2]) <= dz / 2)
        counts[i] = inside.sum()
    return counts


def coverage_by_distance(points, boxes):
    """Fraction of boxes with >= 1 point, per distance bin. Returns list of (hit, total)."""
    counts = points_in_boxes_count(points, boxes) if len(boxes) else np.zeros(0)
    dist = np.linalg.norm(boxes[:, :2], axis=1) if len(boxes) else np.zeros(0)
    out = []
    for lo, hi in zip(DIST_BINS[:-1], DIST_BINS[1:]):
        m = (dist >= lo) & (dist < hi)
        out.append(((counts[m] > 0).sum(), m.sum()))
    return out


def load_livox(dataset, info):
    raw = np.load(os.path.join(dataset.data_path, info['sensor']['livox_path']))
    return np.stack([raw['x'], raw['y'], raw['z']], -1).astype(np.float32)


def plot_bev(path, sensor_pts, livox_pts, boxes, pc_range, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 10))
    if livox_pts is not None:
        ax.scatter(livox_pts[:, 0], livox_pts[:, 1], s=0.2, c='0.75', label='Livox (label sensor)')
    ax.scatter(sensor_pts[:, 0], sensor_pts[:, 1], s=0.3, c='tab:blue', label='this sensor (in Livox frame)')
    for x, y, z, dx, dy, dz, h in boxes[:, :7]:
        corners = np.array([[dx, dy], [dx, -dy], [-dx, -dy], [-dx, dy], [dx, dy]]) / 2
        rot = np.array([[np.cos(h), -np.sin(h)], [np.sin(h), np.cos(h)]])
        c = corners @ rot.T + [x, y]
        ax.plot(c[:, 0], c[:, 1], c='tab:red', lw=1)
    ax.add_patch(plt.Rectangle((pc_range[0], pc_range[1]), pc_range[3] - pc_range[0], pc_range[4] - pc_range[1],
                               fill=False, ls='--', ec='k', lw=1))
    ax.set_xlim(pc_range[0] - 5, pc_range[3] + 5)
    ax.set_ylim(pc_range[1] - 5, pc_range[4] + 5)
    ax.set_aspect('equal')
    ax.legend(loc='upper right', markerscale=20)
    ax.set_title(title)
    fig.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--cfg_file', required=True)
    parser.add_argument('--split', choices=['train', 'test'], default='train')
    parser.add_argument('--num_frames', type=int, default=20)
    parser.add_argument('--plot_dir', type=str, default=None, help='write a BEV png per sampled frame here')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)
    logger = common_utils.create_logger()

    dataset = dataset_classes[cfg.DATA_CONFIG.DATASET](
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, root_path=None,
        training=(args.split == 'train'), logger=logger)
    n = len(dataset.infos)
    if n == 0:
        logger.error('No frames found. Check DATA_PATH (%s) and ImageSets/final/%s.txt'
                     % (dataset.root_path, dataset.split))
        sys.exit(1)
    logger.info('%d frames in split "%s"; sampling %d' % (n, dataset.split, min(n, args.num_frames)))

    rng = np.random.default_rng(0)
    idxs = sorted(rng.choice(n, size=min(n, args.num_frames), replace=False).tolist())
    pc_range = dataset.point_cloud_range
    failures = []
    n_pts, n_pts_in_range, feats = [], [], []
    cov_sensor = np.zeros((len(DIST_BINS) - 1, 2), dtype=np.int64)
    cov_livox = np.zeros_like(cov_sensor)
    if args.plot_dir:
        Path(args.plot_dir).mkdir(parents=True, exist_ok=True)

    for k, i in enumerate(idxs):
        info = dataset.infos[i]
        try:
            _, pts_list, _ = dataset.get_infos_and_points([i])
        except Exception as e:  # report and keep going so one bad frame doesn't hide the rest
            failures.append('frame %d (%s): %r' % (i, info.get('sample_idx'), e))
            continue
        pts = pts_list[0][:-1]  # drop the sentinel point
        if not np.isfinite(pts).all():
            failures.append('frame %d: non-finite values in points' % i)
        in_range = common_utils.mask_points_by_range(pts, pc_range)
        n_pts.append(len(pts))
        n_pts_in_range.append(int(in_range.sum()))
        feats.append(pts[:, 3])

        boxes = info['annos']['gt_boxes_livox'] if 'annos' in info else np.zeros((0, 7))
        boxes = boxes[np.linalg.norm(boxes[:, :2], axis=1) < DIST_BINS[-1]] if len(boxes) else boxes
        cov_sensor += np.array(coverage_by_distance(pts, boxes))
        livox_pts = None
        if dataset.sensor_name != 'livox' and 'livox_path' in info['sensor']:
            try:
                livox_pts = load_livox(dataset, info)
                cov_livox += np.array(coverage_by_distance(livox_pts, boxes))
            except Exception as e:
                logger.warning('Could not load Livox reference for frame %d: %r' % (i, e))

        if args.plot_dir:
            try:
                plot_bev(os.path.join(args.plot_dir, 'frame_%03d_%s.png' % (k, info.get('sample_idx'))),
                         pts, livox_pts, boxes, pc_range,
                         '%s | %s | %d pts' % (dataset.sensor_name, info.get('sequence_name'), len(pts)))
            except ImportError:
                logger.warning('matplotlib not installed; skipping plots')
                args.plot_dir = None

    if n_pts:
        f = np.concatenate(feats)
        logger.info('Points per frame: median %d, min %d; inside POINT_CLOUD_RANGE: median %d, min %d'
                    % (np.median(n_pts), np.min(n_pts), np.median(n_pts_in_range), np.min(n_pts_in_range)))
        logger.info('4th channel after transform: min %.3f  p1 %.3f  p50 %.3f  p99 %.3f  max %.3f'
                    % (f.min(), *np.percentile(f, [1, 50, 99]), f.max()))
        n_empty = int(np.sum(np.array(n_pts_in_range) == 0))
        if n_empty > len(n_pts_in_range) // 2:
            # Most frames empty points to a wrong path, transform or range, not to a sparse sensor.
            failures.append('%d/%d sampled frames have no points inside POINT_CLOUD_RANGE' % (n_empty, len(n_pts_in_range)))
        elif n_empty:
            logger.warning('%d/%d sampled frames have no points inside POINT_CLOUD_RANGE (normal for radar, which '
                           'sometimes records no returns)' % (n_empty, len(n_pts_in_range)))
        if np.abs(f).max() > 6e4:
            logger.warning('4th channel exceeds the fp16 range; use INTENSITY_TRANSFORM: log1p or --amp bf16')

    logger.info('Boxes containing >= 1 point, by distance from the Livox (hit / total):')
    for b, (lo, hi) in enumerate(zip(DIST_BINS[:-1], DIST_BINS[1:])):
        line = '  %2d-%2dm  %-8s %4d / %-4d' % (lo, hi, dataset.sensor_name, *cov_sensor[b])
        if cov_livox[b, 1]:
            line += '   livox %4d / %-4d' % tuple(cov_livox[b])
        logger.info(line)
    near_s = cov_sensor[0, 0] / max(cov_sensor[0, 1], 1)
    near_l = cov_livox[0, 0] / max(cov_livox[0, 1], 1)
    if cov_livox[0, 1] and near_s < 0.5 * near_l:
        logger.warning('Near-range box coverage for %s (%.0f%%) is far below Livox (%.0f%%): the sensor -> Livox '
                       'transform is probably wrong. Look at the BEV plots (--plot_dir).'
                       % (dataset.sensor_name, 100 * near_s, 100 * near_l))

    if failures:
        logger.error('%d problem(s):\n  %s' % (len(failures), '\n  '.join(failures)))
        sys.exit(1)
    logger.info('Preflight passed.')


if __name__ == '__main__':
    main()
