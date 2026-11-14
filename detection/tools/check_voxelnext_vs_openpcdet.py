"""
Check that this repo's VoxelNeXt port computes exactly what OpenPCDet's VoxelNeXt computes.

Builds OpenPCDet's VoxelResBackBone8xVoxelNeXt + VoxelNeXtHead and ours from the same config (voxelnext.yaml),
copies our weights into OpenPCDet's (strict: every parameter name and shape must match), feeds both the same random
voxels and GT boxes, and compares, in train mode, backbone output, head outputs and every loss term, and in eval
mode, the decoded boxes, scores and labels after NMS.

Our head gets the GT boxes with DSERT-RoLL's two extra (zero) velocity columns, OpenPCDet's head without them,
so the check also covers that our head drops them.

Needs a GPU and this repo's environment. OpenPCDet is not installed: --ref_root points at an OpenPCDet source checkout,
and only the VoxelNeXt files and what they import are loaded from it (plus its iou3d_nms CUDA op, built in place).
check_voxelnext.sbatch does the checkout, the build and this run:

    sbatch detection/tools/scripts/snellius/check_voxelnext.sbatch

By hand:
    cd detection/tools
    python check_voxelnext_vs_openpcdet.py --ref_root $HOME/.cache/openpcdet_ref   # exits 1 on any mismatch
"""
import argparse
import copy
import importlib
import os
import sys
import types

import numpy as np
import torch
import yaml
from easydict import EasyDict


def import_reference(ref_root):
    """
    OpenPCDet's VoxelNeXt backbone and head, loaded from the source tree without running pcdet's package __init__
    files (those import every model in OpenPCDet and need all its compiled ops). The packages on the import path are
    replaced by empty stand-ins that point at the real folders; roiaware_pool3d, which box_utils imports but VoxelNeXt
    never calls, is an empty stand-in. pcdet/ops/iou3d_nms/iou3d_nms_cuda*.so must have been built in place.
    """
    pkg = os.path.join(os.path.abspath(ref_root), 'pcdet')
    assert os.path.isdir(pkg), 'no pcdet/ folder in %s' % ref_root

    def stub(name, *rel):
        m = types.ModuleType(name)
        m.__path__ = [os.path.join(pkg, *rel)]
        sys.modules[name] = m
        parent, _, child = name.rpartition('.')
        if parent:
            setattr(sys.modules[parent], child, m)
        return m

    stub('pcdet')
    stub('pcdet.models', 'models')
    stub('pcdet.models.backbones_3d', 'models', 'backbones_3d')
    stub('pcdet.models.dense_heads', 'models', 'dense_heads')
    stub('pcdet.ops', 'ops')
    stub('pcdet.ops.roiaware_pool3d', 'ops', 'roiaware_pool3d')
    sys.modules['pcdet.ops.roiaware_pool3d.roiaware_pool3d_utils'] = types.ModuleType('roiaware_pool3d_utils')
    try:
        import SharedArray  # noqa: F401  (pcdet.utils.common_utils imports it; VoxelNeXt never uses it)
    except ImportError:
        sys.modules['SharedArray'] = types.ModuleType('SharedArray')

    bb = importlib.import_module('pcdet.models.backbones_3d.spconv_backbone_voxelnext')
    hd = importlib.import_module('pcdet.models.dense_heads.voxelnext_head')
    print('reference: OpenPCDet source at', ref_root)
    return bb.VoxelResBackBone8xVoxelNeXt, hd.VoxelNeXtHead


def load_cfg(path):
    with open(path) as f:
        cfg = EasyDict(yaml.safe_load(f))
    pc_range = np.array(cfg.DATA_CONFIG.POINT_CLOUD_RANGE, dtype=np.float32)
    voxel = [p for p in cfg.DATA_CONFIG.DATA_PROCESSOR if p.NAME == 'transform_points_to_voxels_placeholder'][0]
    voxel_size = np.array(voxel.VOXEL_SIZE, dtype=np.float32)
    grid_size = np.round((pc_range[3:6] - pc_range[0:3]) / voxel_size).astype(np.int64)
    return cfg, pc_range, voxel_size, grid_size


def random_batch(grid_size, pc_range, voxel_size, batch_size, n_vox, n_gt, n_cls, gen):
    coords = []
    for b in range(batch_size):
        x = torch.randint(0, int(grid_size[0]), (n_vox,), generator=gen)
        y = torch.randint(0, int(grid_size[1]), (n_vox,), generator=gen)
        z = torch.randint(0, int(grid_size[2]), (n_vox,), generator=gen)
        c = torch.stack([torch.full_like(x, b), z, y, x], 1)
        coords.append(torch.unique(c, dim=0))
    coords = torch.cat(coords).int()
    feats = torch.randn(coords.shape[0], 4, generator=gen)

    lo = torch.tensor(pc_range[:3]) + 5
    hi = torch.tensor(pc_range[3:6]) - 5
    gt = torch.zeros(batch_size, n_gt, 8)
    for b in range(batch_size):
        n = n_gt - b * 2  # frames with different box counts, so zero padding is exercised
        xyz = lo + (hi - lo) * torch.rand(n, 3, generator=gen)
        xyz[:, 2] = -1.0 + torch.rand(n, generator=gen)
        dims = torch.tensor([4.5, 1.9, 1.6]) * (0.6 + 0.8 * torch.rand(n, 3, generator=gen))
        rot = (torch.rand(n, 1, generator=gen) - 0.5) * 2 * np.pi
        cls = torch.randint(1, n_cls + 1, (n, 1), generator=gen).float()
        gt[b, :n] = torch.cat([xyz, dims, rot, cls], 1)
    return feats, coords, gt


def compare(name, a, b, failures, atol=1e-4, rtol=1e-4):
    a = a.detach().float().cpu()
    b = b.detach().float().cpu()
    if a.shape != b.shape:
        failures.append('%s: shape %s vs %s' % (name, tuple(a.shape), tuple(b.shape)))
        print('  FAIL %-34s shape %s vs %s' % (name, tuple(a.shape), tuple(b.shape)))
        return
    diff = (a - b).abs().max().item() if a.numel() else 0.0
    ok = torch.allclose(a, b, atol=atol, rtol=rtol)
    print('  %s %-34s max|diff| %.2e  %s' % ('ok  ' if ok else 'FAIL', name, diff, tuple(a.shape)))
    if not ok:
        failures.append('%s: max diff %.3e' % (name, diff))


def boxes_in_both(po, pr, tol=1e-3):
    """How many of po's boxes (box, score, label) also appear in pr, in any order."""
    def rows(p):
        return torch.cat([p['pred_boxes'].float(), p['pred_scores'].float()[:, None],
                          p['pred_labels'].float()[:, None]], 1)
    a, b = rows(po), rows(pr)
    if len(a) == 0 or len(b) == 0:
        return 0, len(a)
    d = (a[:, None, :] - b[None, :, :]).abs().amax(-1)
    return int((d.min(1).values < tol).sum()), len(a)


def sparse_dense(t):
    """Features of a sparse tensor ordered by (batch, spatial index), so two tensors compare position by position."""
    idx = t.indices.long()
    key = idx[:, 0]
    for d in range(1, idx.shape[1]):
        key = key * (int(t.spatial_shape[d - 1]) + 1) + idx[:, d]
    order = torch.argsort(key)
    return idx[order], t.features[order]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', default='cfgs/det_model_cfgs/dsert/voxelnext.yaml')
    ap.add_argument('--trials', type=int, default=2)
    ap.add_argument('--voxels', type=int, default=30000, help='random voxels per frame')
    ap.add_argument('--gt', type=int, default=12, help='GT boxes in the first frame')
    ap.add_argument('--ref_root', default=None,
                    help='OpenPCDet source checkout; default: an installed, importable pcdet')
    args = ap.parse_args()

    if args.ref_root:
        RefBackbone, RefHead = import_reference(args.ref_root)
    else:
        from pcdet.models.backbones_3d.spconv_backbone_voxelnext import VoxelResBackBone8xVoxelNeXt as RefBackbone
        from pcdet.models.dense_heads.voxelnext_head import VoxelNeXtHead as RefHead
    from al3d_det.models.modules.backbone_3d.voxelnext_backbone import VoxelResBackBone8xVoxelNeXt as OurBackbone
    from al3d_det.models.modules.dense_heads.voxelnext_head import VoxelNeXtHead as OurHead
    import spconv.pytorch as spconv

    cfg, pc_range, voxel_size, grid_size = load_cfg(args.cfg)
    class_names = cfg.CLASS_NAMES
    head_kw = dict(input_channels=cfg.MODEL.BACKBONE_3D.OUT_CHANNEL, num_class=len(class_names),
                   class_names=class_names, grid_size=grid_size, point_cloud_range=pc_range, voxel_size=voxel_size,
                   predict_boxes_when_training=False)
    print('grid', grid_size.tolist(), 'range', pc_range.tolist(), 'voxel', voxel_size.tolist())

    torch.manual_seed(0)
    ours_bb = OurBackbone(copy.deepcopy(cfg.MODEL.BACKBONE_3D), input_channels=4, grid_size=grid_size).cuda()
    ours_hd = OurHead(copy.deepcopy(cfg.MODEL.DENSE_HEAD), **head_kw).cuda()
    ref_bb = RefBackbone(copy.deepcopy(cfg.MODEL.BACKBONE_3D), input_channels=4, grid_size=grid_size).cuda()
    ref_hd = RefHead(copy.deepcopy(cfg.MODEL.DENSE_HEAD), **head_kw).cuda()

    # strict=True: fails loudly if any parameter or buffer name/shape differs from OpenPCDet's
    ref_bb.load_state_dict(ours_bb.state_dict(), strict=True)
    ref_hd.load_state_dict(ours_hd.state_dict(), strict=True)
    print('weights: %d backbone + %d head tensors, names and shapes identical to OpenPCDet'
          % (len(ours_bb.state_dict()), len(ours_hd.state_dict())))

    failures = []
    gen = torch.Generator().manual_seed(1)
    for trial in range(args.trials):
        feats, coords, gt8 = random_batch(grid_size, pc_range, voxel_size, 2, args.voxels, args.gt,
                                          len(class_names), gen)
        # DSERT-RoLL layout for our head: 7 box values, 2 zero velocity columns, class
        gt10 = torch.cat([gt8[..., :7], torch.zeros_like(gt8[..., :2]), gt8[..., 7:]], -1)

        for mode in ('train', 'eval'):
            print('trial %d, %s mode' % (trial, mode))
            if mode == 'eval':
                # An untrained IoU branch outputs values below -1, where OpenPCDet's score hm^(1-r) * ((iou+1)/2)^r
                # is NaN and ours (clamped) is not. A trained one stays inside [-1, 1], where both are identical,
                # so for the eval comparison fix the IoU output at 0.5 in both models (same weights).
                for hd in (ours_hd, ref_hd):
                    last = hd.heads_list[0].iou[-1]
                    with torch.no_grad():
                        last.weight.zero_()
                        last.bias.fill_(0.5)
            for m in (ours_bb, ours_hd, ref_bb, ref_hd):
                m.train(mode == 'train')
            outs = {}
            for tag, bb, hd, gt in (('ours', ours_bb, ours_hd, gt10), ('ref', ref_bb, ref_hd, gt8)):
                bd = {'batch_size': 2, 'voxel_features': feats.cuda().clone(),
                      'voxel_coords': coords.cuda().clone(), 'gt_boxes': gt.cuda().clone()}
                with torch.set_grad_enabled(mode == 'train'):
                    bd = bb(bd)
                    bd = hd(bd)
                    loss = hd.get_loss() if mode == 'train' else None
                outs[tag] = (bd, hd.forward_ret_dict, loss)

            (bo, fo, lo), (br, fr, lr) = outs['ours'], outs['ref']
            io, fe_o = sparse_dense(bo['encoded_spconv_tensor'])
            ir, fe_r = sparse_dense(br['encoded_spconv_tensor'])
            compare('backbone out indices', io, ir, failures, atol=0, rtol=0)
            compare('backbone out features', fe_o, fe_r, failures)
            for k in fo['pred_dicts'][0]:
                compare('head %s' % k, fo['pred_dicts'][0][k], fr['pred_dicts'][0][k], failures)
            if mode == 'train':
                for k in ('heatmaps', 'target_boxes', 'inds', 'masks'):
                    compare('target %s' % k, fo['target_dicts'][k][0], fr['target_dicts'][k][0], failures)
                compare('loss total', lo[0], lr[0], failures)
                for k in sorted(lr[1]):
                    compare('loss %s' % k, torch.tensor(lo[1].get(k, float('nan'))), torch.tensor(lr[1][k]),
                            failures)
            else:
                # Strict test of decoding + NMS: OpenPCDet's head on exactly our backbone output. With bit-identical
                # inputs both heads must return the same boxes in the same order.
                x = bo['encoded_spconv_tensor']
                same_input = {'batch_size': 2, 'encoded_spconv_tensor': spconv.SparseConvTensor(
                    x.features.clone(), x.indices.clone(), x.spatial_shape, x.batch_size)}
                with torch.no_grad():
                    br_same = ref_hd(same_input)
                for b in range(2):
                    po, pr = bo['final_box_dicts'][b], br_same['final_box_dicts'][b]
                    print('  frame %d, same head input: %d vs %d boxes' % (b, len(po['pred_boxes']),
                                                                         len(pr['pred_boxes'])))
                    for k in ('pred_boxes', 'pred_scores', 'pred_labels'):
                        compare('frame %d %s' % (b, k), po[k], pr[k], failures)
                # Whole model, for information only: the two backbones differ by float rounding (~1e-7, checked
                # above), which can swap the order of boxes whose scores are (nearly) tied. An untrained model with
                # the IoU output fixed has many such ties, so compare these as sets.
                for b in range(2):
                    same, n = boxes_in_both(bo['final_box_dicts'][b], br['final_box_dicts'][b])
                    print('  info frame %d, whole model: %d of %d boxes also in OpenPCDet\'s output '
                          '(ordered by score, the scores match: see above)' % (b, same, n))

    print()
    if failures:
        print('MISMATCH in %d comparisons:' % len(failures))
        for f in failures:
            print('  ' + f)
        sys.exit(1)
    print('OK: the port matches OpenPCDet VoxelNeXt (outputs, targets, losses, decoded boxes).')


if __name__ == '__main__':
    main()
