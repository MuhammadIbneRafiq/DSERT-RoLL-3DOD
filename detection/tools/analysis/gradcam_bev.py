"""
Bird's-eye-view Grad-CAM for the CenterPoint detection head. GPU required.

For a labelled object, the target is the head's heatmap logit for the object's class at the object's centre (the
score that decides whether the object is found). Its gradient with respect to the encoder output
(spatial_features_2d, 512 x H x W at 1/8 resolution) gives per-channel weights; the Grad-CAM map is
ReLU(sum_c weight_c * feature_c). Bright = BEV locations whose features raised that object's score.

Each output PNG shows one frame: the point cloud with labelled boxes (green = detected, red = missed by the full
model) and predicted boxes (blue), plus a Grad-CAM panel for up to --per_frame objects (missed ones first).

    python analysis/gradcam_bev.py --ckpt run.pth --sensor radar --data_root /path/to/dsert_roll \
        --name radar_ft --out_dir gradcam_radar_ft --max_frames 8 --weather Fog
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analysis_common as ac  # noqa: E402


def heatmap_logits(model, batch):
    """Encoder forward with gradients; returns (bev_features (1,C,H,W) with grad, heatmap logits (1,K,H,W))."""
    for name in model.ENCODER_MODULES:
        batch = getattr(model, name)(batch)
    feat = batch['spatial_features_2d']
    head = model.dense_head
    hm = head.heads_list[0].hm(head.shared_conv(feat))
    return feat, hm


def gradcam(feat, score):
    grad, = torch.autograd.grad(score, feat, retain_graph=True)
    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * feat).sum(1))[0]
    return (cam / cam.max().clamp(min=1e-12)).detach().cpu().numpy()


def box_corners(box):
    x, y, _, dx, dy, _, h = box[:7]
    c = np.array([[dx, dy], [dx, -dy], [-dx, -dy], [-dx, dy], [dx, dy]]) / 2
    rot = np.array([[np.cos(h), -np.sin(h)], [np.sin(h), np.cos(h)]])
    return c @ rot.T + [x, y]


def main():
    parser = ac.add_common_args(argparse.ArgumentParser(description=__doc__,
                                                        formatter_class=argparse.RawDescriptionHelpFormatter))
    parser.add_argument('--name', required=True)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--per_frame', type=int, default=3, help='objects with a Grad-CAM panel per frame')
    parser.set_defaults(max_frames=8)
    args = parser.parse_args()
    args.out_dir = os.path.abspath(args.out_dir)
    os.makedirs(args.out_dir, exist_ok=True)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    cfg, dataset, model, logger = ac.setup(args)
    pcr = dataset.point_cloud_range
    extent = [pcr[0], pcr[3], pcr[1], pcr[4]]

    for i in ac.select_frames(dataset, args.max_frames, args.weather):
        info = dataset.infos[i]
        batch = ac.load_frame(dataset, i)
        gt = ac.gt_in_range(batch, pcr)
        if len(gt) == 0:
            continue
        pred = ac.run_model(model, batch)
        _, _, detected = ac.match_detections(pred, gt, cfg.CLASS_NAMES)

        batch = ac.load_frame(dataset, i)  # fresh batch: run_model above filled it with intermediate tensors
        with torch.enable_grad():
            feat, hm = heatmap_logits(model, batch)
            H, W = hm.shape[2:]
            order = sorted(range(len(gt)), key=lambda k: (bool(detected[k]), float(torch.linalg.norm(gt[k, :2]))))
            cams = []
            for k in order[:args.per_frame]:
                col = int(((gt[k, 0] - pcr[0]) / (pcr[3] - pcr[0]) * W).clamp(0, W - 1))
                row = int(((gt[k, 1] - pcr[1]) / (pcr[4] - pcr[1]) * H).clamp(0, H - 1))
                c = int(gt[k, -1]) - 1  # class = last column
                r0, r1, c0, c1 = max(row - 1, 0), min(row + 2, H), max(col - 1, 0), min(col + 2, W)
                score = hm[0, c, r0:r1, c0:c1].max()   # peak logit in the 3x3 cells around the centre
                cams.append((k, gradcam(feat, score), torch.sigmoid(score).item()))

        pts = batch['points'][batch['points'][:, 0] == 0][:, 1:3].cpu().numpy()
        gt_np, det_np = gt.cpu().numpy(), detected.cpu().numpy()
        fig, axes = plt.subplots(1, 1 + len(cams), figsize=(6 * (1 + len(cams)), 9))
        axes = np.atleast_1d(axes)
        for ax_i, ax in enumerate(axes):
            ax.scatter(pts[:, 0], pts[:, 1], s=0.3, c='0.6')
            if ax_i > 0:
                k, cam, prob = cams[ax_i - 1]
                ax.imshow(cam, origin='lower', extent=extent, cmap='jet', alpha=0.55, aspect='auto')
                ax.plot(*box_corners(gt_np[k]).T, c='white', lw=2.5)
                ax.set_title('%s %s, %.0f m, %s\nheatmap score %.2f'
                             % (cfg.CLASS_NAMES[int(gt_np[k, -1]) - 1], 'detected' if det_np[k] else 'MISSED',
                                np.linalg.norm(gt_np[k, :2]), args.sensor, prob))
            else:
                for b in pred['pred_boxes'].cpu().numpy():
                    ax.plot(*box_corners(b).T, c='tab:blue', lw=1)
                ax.set_title('%s | %s | %s/%s\ngreen=detected red=missed blue=predicted'
                             % (args.name, info.get('sequence_name'), info.get('weather'), info.get('light')))
            for b, d in zip(gt_np, det_np):
                ax.plot(*box_corners(b).T, c='tab:green' if d else 'tab:red', lw=1.2)
            ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3]); ax.set_aspect('equal')
        fig.tight_layout()
        out = os.path.join(args.out_dir, '%s_%s_%s.png' % (args.name, info.get('weather'), info.get('sample_idx', i)))
        fig.savefig(out, dpi=90)
        plt.close(fig)
        logger.info('wrote %s' % out)


if __name__ == '__main__':
    main()
