"""
Compare a result.pkl (predictions written by the evaluation) with the ground truth, to see why AP is what it is:
are there predictions at all, which classes and scores, and are they where the labelled objects are?

For every labelled Vehicle it finds the nearest predicted box centre (bird's-eye view) and reports how often one
lies within 1 / 2 / 4 m. It repeats that with the predictions transformed (x/y swapped, y mirrored, scaled by 2 or
1/2): if one of those matches much better than the identity, the boxes come out in the wrong coordinates.

CPU only; run from detection/tools on a login node:
  python diagnose_predictions.py --cfg_file cfgs/det_model_cfgs/dsert/voxelnext.yaml \
      --result_pkl ../output/det_model_cfgs/dsert/voxelnext/vnx_radar_s1/eval/eval_with_train/epoch_20/val/result.pkl \
      --set DATA_CONFIG.DATA_PATH $HOME/data/dsert_roll DATA_CONFIG.SENSOR.NAME radar
"""
import argparse
import pickle
from collections import Counter

import numpy as np

from al3d_utils import common_utils
from al3d_utils.config_utils import cfg, cfg_from_list, cfg_from_yaml_file
from al3d_det.datasets import __all__ as dataset_classes


def q(a, qs=(0, 5, 50, 95, 100)):
    a = np.asarray(a, dtype=float)
    return 'n/a' if a.size == 0 else ' '.join('%.2f' % v for v in np.percentile(a, qs))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cfg_file', required=True)
    p.add_argument('--result_pkl', required=True)
    p.add_argument('--score_min', type=float, default=0.0, help='ignore predictions below this score')
    p.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER)
    a = p.parse_args()
    cfg_from_yaml_file(a.cfg_file, cfg)
    if a.set_cfgs:
        cfg_from_list(a.set_cfgs, cfg)
    logger = common_utils.create_logger()
    ds = dataset_classes[cfg.DATA_CONFIG.DATASET](dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES,
                                                  root_path=None, training=False, logger=logger)
    with open(a.result_pkl, 'rb') as f:
        dets = pickle.load(f)
    print('frames: result.pkl %d, dataset %d' % (len(dets), len(ds.infos)))
    n = min(len(dets), len(ds.infos))
    same_id = sum(str(dets[i].get('frame_id')) == str(ds.infos[i].get('frame_id', dets[i].get('frame_id')))
                  for i in range(n))
    print('frame_id equal at the same index: %d / %d (evaluation pairs them by index)' % (same_id, n))

    n_pred, names, gt_names, scores, pb, gb = [], Counter(), Counter(), [], [], []
    pairs = []  # (gt vehicle boxes, all pred boxes) per frame
    for i in range(n):
        d, gt = dets[i], ds.infos[i]['annos']
        keep = np.asarray(d['score']) >= a.score_min
        boxes = np.asarray(d['boxes_lidar'])[keep][:, :7] if keep.any() else np.zeros((0, 7))
        n_pred.append(len(boxes))
        names.update([str(x) for x in np.asarray(d['name'])[keep]])
        scores.extend(np.asarray(d['score'])[keep].tolist())
        pb.append(boxes)
        gt_names.update([str(x) for x in gt['name']])
        gmask = np.array([str(x).lower() in ('vehicle', 'car') for x in gt['name']], dtype=bool)
        g = np.asarray(gt['gt_boxes_livox'])[gmask][:, :7] if gmask.any() else np.zeros((0, 7))
        g = g[np.linalg.norm(g[:, :2], axis=1) < 70.5] if len(g) else g
        gb.append(g)
        pairs.append((g, boxes))

    P = np.concatenate(pb) if pb else np.zeros((0, 7))
    G = np.concatenate(gb) if gb else np.zeros((0, 7))
    print('\npredictions per frame (min p5 median p95 max):', q(n_pred))
    print('predicted classes:', dict(names))
    print('scores:', q(scores))
    print('GT classes:', dict(gt_names))
    print('GT vehicles (< 70.5 m):', len(G))
    for name, B in (('pred', P), ('GT  ', G)):
        if len(B):
            print('%s x %s | y %s | z %s' % (name, q(B[:, 0]), q(B[:, 1]), q(B[:, 2])))
            print('%s dx %s | dy %s | dz %s | heading %s' % (name, q(B[:, 3]), q(B[:, 4]), q(B[:, 5]), q(B[:, 6])))

    transforms = {
        'identity': lambda b: b[:, :2],
        'swap x/y': lambda b: b[:, [1, 0]],
        'mirror y': lambda b: b[:, :2] * [1, -1],
        'x2': lambda b: b[:, :2] * 2,
        'x0.5': lambda b: b[:, :2] * 0.5,
    }
    print('\nnearest predicted centre to each GT vehicle (BEV); share within 1 / 2 / 4 m:')
    for tname, t in transforms.items():
        dists = []
        for g, b in pairs:
            if len(g) == 0:
                continue
            if len(b) == 0:
                dists.extend([np.inf] * len(g))
                continue
            c = t(b)
            dists.extend(np.min(np.linalg.norm(g[:, None, :2] - c[None], axis=2), axis=1).tolist())
        dists = np.asarray(dists)
        if dists.size:
            print('  %-9s %5.1f%% %5.1f%% %5.1f%%   median %.2f m'
                  % (tname, *(100 * np.mean(dists < r) for r in (1, 2, 4)), np.median(dists)))

    for i in range(n):
        g, b = pairs[i]
        if len(g):
            order = np.argsort(-np.asarray(dets[i]['score']))[:5]
            print('\nexample frame %s: %d GT vehicles, %d predictions' % (dets[i].get('frame_id'), len(g), len(b)))
            print('  GT   (x y z dx dy dz heading):', np.round(g[:3], 2).tolist())
            print('  pred top-5 by score:', [(str(dets[i]['name'][k]), round(float(dets[i]['score'][k]), 3),
                                              np.round(np.asarray(dets[i]['boxes_lidar'])[k][:7], 2).tolist())
                                             for k in order])
            break


if __name__ == '__main__':
    main()
