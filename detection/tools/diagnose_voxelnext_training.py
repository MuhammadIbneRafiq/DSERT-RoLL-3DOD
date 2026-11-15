"""
Look inside a trained VoxelNeXt checkpoint on a few real frames, to see why it does or does not detect anything.

Per frame it prints: points, sparse voxels reaching the head, labelled boxes, how many got a training target,
how far each box centre is from the voxel it was assigned to, the predicted heatmap there versus elsewhere, and
whether the regression target decodes back to the labelled centre (a coordinate-convention check). Then the
individual loss terms on the same batch.

1 GPU, a few minutes. From detection/tools, in the training environment:
  python diagnose_voxelnext_training.py --cfg_file cfgs/det_model_cfgs/dsert/voxelnext.yaml \
      --ckpt ../output/det_model_cfgs/dsert/voxelnext/vnx_radar_s1/ckpt/checkpoint_epoch_20.pth \
      --set DATA_CONFIG.DATA_PATH $HOME/data/dsert_roll DATA_CONFIG.SENSOR.NAME radar
"""
import argparse
import copy

import numpy as np
import torch

from al3d_utils import common_utils
from al3d_utils.config_utils import cfg, cfg_from_list, cfg_from_yaml_file
from al3d_utils.model_utils import load_params_from_file
from al3d_det.datasets import build_dataloader
from al3d_det.models import build_network, load_data_to_gpu


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cfg_file', required=True)
    p.add_argument('--ckpt', required=True)
    p.add_argument('--split', choices=['train', 'test'], default='train')
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--batches', type=int, default=3)
    p.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER)
    a = p.parse_args()
    cfg_from_yaml_file(a.cfg_file, cfg)
    if a.set_cfgs:
        cfg_from_list(a.set_cfgs, cfg)
    logger = common_utils.create_logger()
    torch.manual_seed(0)
    np.random.seed(0)

    ds, loader, _ = build_dataloader(dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES,
                                     batch_size=a.batch_size, dist=False, workers=2, logger=logger,
                                     training=(a.split == 'train'))
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=ds)
    load_params_from_file(model, a.ckpt, logger=logger)
    model.cuda()
    head = model.dense_head
    stride = head.feature_map_stride
    vs = head.voxel_size
    pcr = head.point_cloud_range
    print('grid_size %s, voxel_size %s, range %s, head stride %s, class_names %s'
          % (ds.grid_size, list(vs), list(pcr), stride, head.class_names))

    for bi, batch_cpu in enumerate(loader):
        if bi >= a.batches:
            break
        batch = copy.deepcopy(batch_cpu)
        load_data_to_gpu(batch)
        gt = batch['gt_boxes']
        if bi == 0:
            print('gt_boxes tensor %s; first row %s; label values %s'
                  % (tuple(gt.shape), gt[0, 0].cpu().numpy().round(2).tolist(),
                     torch.unique(gt[..., -1]).cpu().numpy().tolist()))

        model.eval()
        with torch.no_grad():
            model(batch)
        x = batch['encoded_spconv_tensor']
        spatial_shape, batch_index, voxel_indices, spatial_indices, num_voxels = head._get_voxel_infos(x)
        td = head.assign_targets(gt, num_voxels, spatial_indices, spatial_shape, batch_index)
        hm = head.forward_ret_dict['pred_dicts'][0]['hm'].sigmoid()  # (N, C), eval-mode BN, tensor voxel order
        hm_t = td['heatmaps'][0]
        print('  head voxels grouped by frame: %s' % bool((batch_index[1:] >= batch_index[:-1]).all()))

        for b in range(gt.shape[0]):
            n_pts = int((batch['points'][:, 0] == b).sum())
            labels = gt[b, :, -1]
            n_gt = int((labels > 0).sum())
            m = td['masks'][0][b].bool()
            inds = td['inds'][0][b][m]
            tb = td['target_boxes'][0][b][m]
            ks = torch.nonzero(m).squeeze(1)
            boxes = td['gt_boxes'][0][b][ks]  # target k was written for box k of this head
            line = 'batch %d frame %d: %6d points, %5d voxels, %3d boxes, %3d targets' % (
                bi, b, n_pts, num_voxels[b], n_gt, int(m.sum()))
            if m.any():
                dist_m = tb[:, 0:2].norm(dim=1) * stride * vs[0]
                si = spatial_indices[b][inds].float()
                dec_x = (si[:, 0] + tb[:, 0]) * stride * vs[0] + pcr[0]
                dec_y = (si[:, 1] + tb[:, 1]) * stride * vs[1] + pcr[1]
                dec_err = torch.max((dec_x - boxes[:, 0]).abs().max(), (dec_y - boxes[:, 1]).abs().max())
                rows_b = torch.nonzero(batch_index == b).squeeze(1)  # this frame's voxels, in tensor order
                gi = rows_b[inds]
                hm_pos = hm[gi].max(dim=1).values
                hm_b = hm[rows_b]
                line += (' | box-to-voxel m: median %.2f max %.2f | decode err %.3f m | target hm at those voxels '
                         'min %.2f | pred hm there median %.3f, elsewhere p99 %.3f max %.3f'
                         % (dist_m.median(), dist_m.max(), dec_err, hm_t[gi].max(dim=1).values.min(),
                            hm_pos.median(), torch.quantile(hm_b.max(dim=1).values.float(), 0.99), hm_b.max()))
            print(line)

        model.train()
        batch = copy.deepcopy(batch_cpu)
        load_data_to_gpu(batch)
        with torch.no_grad():
            _, tb_dict, _ = model(batch)
        print('  losses (train mode, this batch):',
              {k: round(float(v), 4) for k, v in tb_dict.items() if 'loss' in k})


if __name__ == '__main__':
    main()
