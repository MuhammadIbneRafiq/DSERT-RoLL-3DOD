"""
Extract one encoder feature vector per labelled object, for t-SNE and probing (plot_tsne.py). GPU required.

For every ground-truth box in the sampled validation frames it stores the encoder's BEV feature at the box centre
(512 channels of spatial_features_2d, the representation that is transferred between runs) together with:
class, weather, light, distance, number of sensor points inside the box, and whether the full model detected it
(best same-class IoU against its predictions, at the evaluation's IoU thresholds).

    python analysis/extract_features.py --ckpt run.pth --sensor radar --data_root /path/to/dsert_roll \
        --name radar_ft --out features_radar_ft.npz
"""
import argparse
import os
import sys

import numpy as np
import torch
import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analysis_common as ac  # noqa: E402


def main():
    parser = ac.add_common_args(argparse.ArgumentParser(description=__doc__,
                                                        formatter_class=argparse.RawDescriptionHelpFormatter))
    parser.add_argument('--name', required=True, help='label for this model in the plots, e.g. radar_ft')
    parser.add_argument('--out', required=True, help='output .npz')
    args = parser.parse_args()
    args.out = os.path.abspath(args.out)

    cfg, dataset, model, logger = ac.setup(args)
    pc_range = dataset.point_cloud_range
    frames = ac.select_frames(dataset, args.max_frames, args.weather)
    logger.info('%s: %d frames, sensor=%s' % (args.name, len(frames), args.sensor))

    rows = {k: [] for k in ['feat', 'cls', 'dist', 'npts', 'iou', 'score', 'detected', 'frame']}
    meta = {k: [] for k in ['weather', 'light', 'sequence']}
    for i in tqdm.tqdm(frames, desc='frames'):
        batch = ac.load_frame(dataset, i)
        gt = ac.gt_in_range(batch, pc_range)
        if len(gt) == 0:
            continue
        pred = ac.run_model(model, batch)
        feat = ac.sample_bev(batch['spatial_features_2d'][0].float(), gt[:, :2], pc_range)
        iou, score, detected = ac.match_detections(pred, gt, cfg.CLASS_NAMES)
        npts = ac.points_in_boxes(batch['points'][batch['points'][:, 0] == 0][:, 1:4], gt[:, :7])

        rows['feat'].append(feat.cpu().numpy().astype(np.float32))
        rows['cls'].append(gt[:, 7].long().cpu().numpy())
        rows['dist'].append(torch.linalg.norm(gt[:, :2], dim=1).cpu().numpy())
        rows['npts'].append(npts.cpu().numpy())
        rows['iou'].append(iou.cpu().numpy())
        rows['score'].append(score.cpu().numpy())
        rows['detected'].append(detected.cpu().numpy())
        rows['frame'].append(np.full(len(gt), i))
        info = dataset.infos[i]
        for key, src in [('weather', 'weather'), ('light', 'light'), ('sequence', 'sequence_name')]:
            meta[key].append(np.array([str(info.get(src, 'unknown'))] * len(gt)))

    out = {k: np.concatenate(v) for k, v in rows.items()}
    out.update({k: np.concatenate(v) for k, v in meta.items()})
    out['class_names'] = np.array(cfg.CLASS_NAMES)
    out['name'], out['sensor'], out['ckpt'] = np.array(args.name), np.array(args.sensor), np.array(args.ckpt)
    np.savez_compressed(args.out, **out)

    n = len(out['cls'])
    logger.info('saved %d objects to %s' % (n, args.out))
    for c, cname in enumerate(cfg.CLASS_NAMES, start=1):
        m = out['cls'] == c
        if m.any():
            logger.info('  %-10s %5d objects, recall %.3f' % (cname, m.sum(), out['detected'][m].mean()))


if __name__ == '__main__':
    main()
