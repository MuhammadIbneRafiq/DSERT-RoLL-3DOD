"""
Can VoxelNeXt memorise a few real frames? A model that trains correctly fits 8 fixed frames within a few hundred
steps: the heatmap at the labelled objects rises towards 1 while it stays low elsewhere. Runs the same frames under
several training setups, so the one that fails points at the cause:

  fp32  + adam_onecycle wrapper (the config's optimizer, at a constant LR)
  fp16  + adam_onecycle wrapper + GradScaler (what train_mm.py does with --amp fp16)
  fp32  + plain torch.optim.Adam

Also prints how many model parameters the optimizer actually holds, gradient norms of encoder vs head, and the
GradScaler scale (a scale that keeps falling means steps are being skipped).

1 GPU, ~10 min. From detection/tools, in the training environment:
  python overfit_voxelnext.py --cfg_file cfgs/det_model_cfgs/dsert/voxelnext.yaml \
      --set DATA_CONFIG.DATA_PATH $HOME/data/dsert_roll DATA_CONFIG.SENSOR.NAME radar
"""
import argparse
import copy

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from al3d_utils import common_utils
from al3d_utils.config_utils import cfg, cfg_from_list, cfg_from_yaml_file
from al3d_utils.optimize_utils import build_optimizer
from al3d_det.datasets import build_dataloader
from al3d_det.models import build_network, load_data_to_gpu


def grad_norm(params):
    g = [p.grad.detach().float().norm() for p in params if p.grad is not None]
    return float(torch.stack(g).norm()) if g else 0.0


def hm_report(model, batches):
    """Mean predicted heatmap (eval mode) at the voxels that carry a target, vs the 99th percentile elsewhere."""
    model.eval()
    head = model.dense_head
    at_obj, other = [], []
    with torch.no_grad():
        for bc in batches:
            b = copy.deepcopy(bc)
            load_data_to_gpu(b)
            model(b)
            x = b['encoded_spconv_tensor']
            spatial_shape, _, _, spatial_indices, num_voxels = head._get_voxel_infos(x)
            td = head.assign_targets(b['gt_boxes'], num_voxels, spatial_indices, spatial_shape)
            hm = head.forward_ret_dict['pred_dicts'][0]['hm'].sigmoid().max(dim=1).values
            pos = td['heatmaps'][0].max(dim=1).values >= 1.0
            at_obj.append(hm[pos])
            other.append(hm[~pos])
    model.train()
    a, o = torch.cat(at_obj), torch.cat(other)
    return float(a.mean()) if a.numel() else float('nan'), float(torch.quantile(o.float(), 0.99))


def run(setup, batches, ds, steps, lr):
    torch.manual_seed(0)
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=ds).cuda()
    if setup['optim'] == 'onecycle':
        optimizer = build_optimizer(model, cfg.OPTIMIZATION)
        optimizer.lr = lr
        groups = optimizer.opt.param_groups
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99))
        groups = optimizer.param_groups
    in_opt = {id(p) for g in groups for p in g['params']}
    model_params = [p for p in model.parameters() if p.requires_grad]
    missing = [n for n, p in model.named_parameters() if p.requires_grad and id(p) not in in_opt]
    print('\n==== %s: optimizer holds %d of %d parameter tensors%s'
          % (setup['name'], sum(id(p) in in_opt for p in model_params), len(model_params),
             ('; MISSING e.g. %s' % missing[:5]) if missing else ''))

    scaler = None
    if setup['amp']:
        model.set_amp(torch.float16)
        scaler = torch.amp.GradScaler('cuda')
    enc = [p for n, p in model.named_parameters() if n.startswith(('vfe', 'backbone3d'))]
    head = [p for n, p in model.named_parameters() if n.startswith('dense_head')]

    print('step   loss   hm_loss  loc_loss | grad enc / head | scale   | hm at objects / elsewhere p99')
    a, o = hm_report(model, batches)
    print('%4d  %s | %.3f / %.3f' % (0, ' ' * 56, a, o))
    model.train()
    for step in range(1, steps + 1):
        b = copy.deepcopy(batches[step % len(batches)])
        load_data_to_gpu(b)
        optimizer.zero_grad()
        ret, tb, _ = model(b)
        loss = ret['loss'].mean()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            ge, gh = grad_norm(enc), grad_norm(head)
            clip_grad_norm_(model.parameters(), cfg.OPTIMIZATION.GRAD_NORM_CLIP)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            ge, gh = grad_norm(enc), grad_norm(head)
            clip_grad_norm_(model.parameters(), cfg.OPTIMIZATION.GRAD_NORM_CLIP)
            optimizer.step()
        if step % 50 == 0 or step == 1:
            a, o = hm_report(model, batches)
            print('%4d  %6.3f  %6.3f  %6.3f   | %8.3f / %8.3f | %7s | %.3f / %.3f'
                  % (step, float(loss), tb.get('hm_loss_head_0', np.nan), tb.get('loc_loss_head_0', np.nan),
                     ge, gh, '%.0f' % scaler.get_scale() if scaler else '-', a, o))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cfg_file', required=True)
    p.add_argument('--frames', type=int, default=8)
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--steps', type=int, default=400)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--only', default='', help='comma-separated subset of: fp32_onecycle,fp16_onecycle,fp32_adam')
    p.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER)
    a = p.parse_args()
    cfg_from_yaml_file(a.cfg_file, cfg)
    if a.set_cfgs:
        cfg_from_list(a.set_cfgs, cfg)
    logger = common_utils.create_logger()

    # Evaluation-split loader: no augmentation and a fixed order, so the same frames come back every step.
    ds, loader, _ = build_dataloader(dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES,
                                     batch_size=a.batch_size, dist=False, workers=2, logger=logger, training=False)
    batches = []
    for bc in loader:
        if 'gt_boxes' not in bc:
            raise SystemExit('the evaluation loader gives no gt_boxes; cannot run this check')
        if (bc['gt_boxes'][..., -1] > 0).sum(axis=1).min() == 0:
            continue  # keep batches where every frame has at least one labelled object
        batches.append(bc)
        if len(batches) * a.batch_size >= a.frames:
            break
    print('fixed frames: %d (in %d batches), labelled objects: %d'
          % (len(batches) * a.batch_size, len(batches), sum(int((bc['gt_boxes'][..., -1] > 0).sum()) for bc in batches)))

    setups = [dict(name='fp32_onecycle', amp=False, optim='onecycle'),
              dict(name='fp16_onecycle', amp=True, optim='onecycle'),
              dict(name='fp32_adam', amp=False, optim='adam')]
    if a.only:
        setups = [s for s in setups if s['name'] in a.only.split(',')]
    for s in setups:
        run(s, batches, ds, a.steps, a.lr)


if __name__ == '__main__':
    main()
