"""
Data smoke test for MAN TruckScenes (CPU only; no GPU, no training). Run after create_truckscenes_infos.py:

    cd detection/tools
    python check_truckscenes_data.py --dataroot /path/to/man-truckscenes --version v1.2-mini
    python check_truckscenes_data.py --dataroot $TRUCKSCENES --version v1.2-trainval --num_samples 30

Checks, each against the official devkit (exit code 1 if any fails):
 [1] points: our loader (all 6 LiDARs x 3 sweeps, all 6 radars x 6 sweeps, in the LIDAR_LEFT keyframe's ego frame)
     equals the devkit's own PointCloud.from_file_multisweep(chan, ref_chan=LIDAR_LEFT) moved from LIDAR_LEFT into
     the ego frame: x, y, z, time lag, LiDAR intensity, radar rcs.
 [2] boxes: the LiDAR / radar points we load (keyframe only) that fall in each GT box (devkit points_in_box, our
     ego-frame boxes) against the annotation's num_lidar_pts / num_radar_pts. Catches any frame or box-convention
     error (a wrong yaw, swapped w/l, the wrong ego pose).
 [3] evaluation: the GT boxes of the val split, written as predictions through TruckScenesDataset.evaluation (ego ->
     global conversion, submission file, devkit DetectionEval detection_cvpr_2024), must score mAP = 1 with zero
     translation / scale / orientation errors.
 [4] training samples: TruckScenesDataset(training=True) items (augmentation, range crop, voxel grid), with load time.
 [5] DynamicMeanVFE voxel keys: int64 keys give the right voxel for every point on the TruckScenes grid at batch 16,
     and the same voxels as the old int32 keys on the DSERT-RoLL grid.
"""
import argparse
import importlib.abc
import importlib.machinery
import importlib.util
import os
import sys
import time
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
for p in (os.path.join(REPO, 'detection'), os.path.join(REPO, 'utils'), os.path.join(REPO, 'truckscenes-devkit', 'src')):
    if p not in sys.path:
        sys.path.insert(0, p)


class _StubCompiledOps(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Without the repo's compiled CUDA ops (a CPU-only machine), import empty modules in their place: nothing this
    script runs calls them."""

    def find_spec(self, name, path, target=None):
        if name.startswith('al3d_utils.ops.') and name.rsplit('.', 1)[-1].endswith(('_cuda', '_ext', '_cpu')):
            return importlib.machinery.ModuleSpec(name, self)
        return None

    def create_module(self, spec):
        return types.ModuleType(spec.name)

    def exec_module(self, module):
        pass


FAILURES = []


def report(ok, msg):
    print('  %s %s' % ('ok  ' if ok else 'FAIL', msg))
    if not ok:
        FAILURES.append(msg)


def check_points(trucksc, tu, infos, dataroot, n):
    from pyquaternion import Quaternion
    from truckscenes.utils.data_classes import LidarPointCloud, RadarPointCloud
    print('[1] our merged points == devkit from_file_multisweep (moved into the ego frame)')
    for modality, nsweeps, cls, feat_name, feat_row in (('lidar', 3, LidarPointCloud, 'intensity', 3),
                                                        ('radar', 6, RadarPointCloud, 'rcs', 6)):
        worst = {'xyz': 0.0, 'time_lag': 0.0, feat_name: 0.0}
        n_pts = []
        for info in infos[:n]:
            sample = trucksc.get('sample', info['sample_token'])
            ours = tu.load_sweeps(dataroot, info['sweeps'][modality], modality, nsweeps, info['time_stamp'], feat_name)
            ref_sd = trucksc.get('sample_data', sample['data'][tu.REF_CHANNEL])
            ref_cs = trucksc.get('calibrated_sensor', ref_sd['calibrated_sensor_token'])
            car_from_ref = tu.transform_matrix(ref_cs['translation'], Quaternion(ref_cs['rotation']))
            parts = []
            for chan in sorted(info['sweeps'][modality]):
                pc, times = cls.from_file_multisweep(trucksc, sample, chan, tu.REF_CHANNEL, nsweeps=nsweeps,
                                                     min_distance=tu.MIN_DISTANCE)
                pc.transform(car_from_ref)
                parts.append(np.stack([pc.points[0], pc.points[1], pc.points[2], pc.points[feat_row],
                                       times.reshape(-1)], 1))
            ref = np.concatenate(parts, 0)
            if ref.shape != ours.shape:
                report(False, '%s %s: %s points ours vs %s devkit' % (modality, info['sample_token'], ours.shape,
                                                                     ref.shape))
                continue
            worst['xyz'] = max(worst['xyz'], float(np.abs(ours[:, :3] - ref[:, :3]).max(initial=0)))
            worst[feat_name] = max(worst[feat_name], float(np.abs(ours[:, 3] - ref[:, 3]).max(initial=0)))
            worst['time_lag'] = max(worst['time_lag'], float(np.abs(ours[:, 4] - ref[:, 4]).max(initial=0)))
            n_pts.append(len(ours))
        report(worst['xyz'] < 1e-3 and worst[feat_name] < 1e-3 and worst['time_lag'] < 1e-4,
               '%s, %d sweeps, %d samples (%.0f points each): max |diff| xyz %.1e m, %s %.1e, time lag %.1e s'
               % (modality, nsweeps, len(n_pts), np.mean(n_pts) if n_pts else 0, worst['xyz'], feat_name,
                  worst[feat_name], worst['time_lag']))


def check_boxes(tu, infos, dataroot, n):
    from pyquaternion import Quaternion
    from truckscenes.utils.data_classes import Box
    from truckscenes.utils.geometry_utils import points_in_box
    print('[2] points in our ego-frame GT boxes vs the annotations\' num_lidar_pts / num_radar_pts (keyframe only)')
    for modality, key in (('lidar', 'num_lidar_pts'), ('radar', 'num_radar_pts')):
        ours_cnt, ann_cnt = [], []
        for info in infos[:n]:
            pts = tu.load_sweeps(dataroot, info['sweeps'][modality], modality, 1, info['time_stamp'], 'none')
            a = info['annos']
            for b, cnt in zip(a['gt_boxes_ego'], a[key]):
                box = Box(b[:3].tolist(), [b[4], b[3], b[5]], Quaternion(axis=[0, 0, 1], radians=b[6]))
                ours_cnt.append(int(points_in_box(box, pts[:, :3].T).sum()))
                ann_cnt.append(int(cnt))
        o, g = np.array(ours_cnt), np.array(ann_cnt)
        m = g >= 10
        corr = float(np.corrcoef(o[m], g[m])[0, 1]) if m.sum() > 2 else float('nan')
        rel = np.abs(o[m] - g[m]) / g[m]
        both_zero = float(((o == 0) == (g == 0)).mean())
        print('       %s: %d boxes; boxes with >= 10 annotated points: %d, correlation %.3f, median |ours-ann|/ann %.3f,'
              ' within 20%%: %.2f; empty/non-empty agree for %.2f of all boxes'
              % (modality, len(g), m.sum(), corr, np.median(rel) if m.any() else float('nan'),
                 (rel < 0.2).mean() if m.any() else float('nan'), both_zero))
        # A frame or convention error gives near-zero overlap; small differences come from the annotators' point
        # counting (e.g. sensor filtering) and are not errors.
        report(m.sum() == 0 or corr > 0.9, '%s point counts follow the annotations (correlation %.3f > 0.9)'
               % (modality, corr))


def build_dataset(cfg_file, dataroot, version, info_dir, training, sensor=None, interval=1):
    from al3d_utils.config_utils import cfg, cfg_from_yaml_file, cfg_from_list
    from al3d_det.datasets import TruckScenesDataset
    cwd = os.getcwd()
    os.chdir(HERE)   # _BASE_CONFIG_ paths are relative to detection/tools
    try:
        cfg_from_yaml_file(cfg_file, cfg)
    finally:
        os.chdir(cwd)
    splits = {'train': 'mini_train', 'test': 'mini_val'} if version.endswith('mini') else {'train': 'train',
                                                                                             'test': 'val'}
    sets = ['DATA_CONFIG.DATA_PATH', dataroot, 'DATA_CONFIG.VERSION', version,
            'DATA_CONFIG.DATA_SPLIT.train', splits['train'], 'DATA_CONFIG.DATA_SPLIT.test', splits['test'],
            'DATA_CONFIG.SAMPLED_INTERVAL.train', str(interval), 'DATA_CONFIG.SAMPLED_INTERVAL.test', str(interval)]
    if info_dir:
        sets += ['DATA_CONFIG.INFO_DIR', info_dir]
    if sensor:
        sets += ['DATA_CONFIG.SENSOR.NAME', sensor]
    cfg_from_list(sets, cfg)

    class Log:
        def info(self, m):
            print('       ' + m)
        warning = info
    return cfg, TruckScenesDataset(cfg.DATA_CONFIG, cfg.CLASS_NAMES, root_path=dataroot, training=training,
                                   logger=Log())


def check_eval(args, out_dir):
    print('[3] GT as predictions -> TruckScenesDataset.evaluation -> devkit DetectionEval: expect mAP 1')
    cfg, ds = build_dataset(args.cfg, args.dataroot, args.version, args.info_dir, training=False)
    det_annos = []
    # Distinct scores: the devkit resamples the TP errors over the score axis (np.interp on the confidences), which is
    # undefined when every prediction has the same score (it then reports the error of a single box for all recalls).
    rng = np.random.RandomState(0)
    attr_hits = {}   # class -> [attribute right, attribute judged], to cross-check the devkit's AAE
    for info in ds.infos:
        a = info['annos']
        keep = (a['name'] != '') & ((a['num_lidar_pts'] + a['num_radar_pts']) > 0)
        boxes = a['gt_boxes_ego'][keep].copy()
        boxes[np.isnan(boxes)] = 0.0
        det_annos.append({'name': a['name'][keep], 'score': rng.uniform(0.5, 1.0, int(keep.sum())),
                          'boxes_lidar': boxes,
                          'frame_id': info['sample_token'], 'sequence_name': info['sequence_name']})
        for name, attr, box in zip(a['name'][keep], a['attribute_name'][keep], boxes):
            if attr:
                pred_attr = ds.default_attribute(name, box[7:9])
                h = attr_hits.setdefault(name, [0, 0])
                h[0] += int(pred_attr == attr)
                h[1] += 1
    result_str, ret = ds.evaluation(det_annos, cfg.CLASS_NAMES, output_path=out_dir)
    print(result_str)
    # The devkit scores a class with no GT in the split as AP 0 (mini_val has e.g. no bus), so judge the classes that
    # have GT within the evaluation ranges, from the devkit's own metrics_summary.json.
    import json
    summary = json.load(open(os.path.join(out_dir, 'truckscenes_eval', 'metrics_summary.json')))['all']
    present = sorted(set(n for d in det_annos for n in d['name']))
    with_gt = [c for c in present if summary['label_tp_errors'][c]['trans_err'] < 1.0 or summary['mean_dist_aps'][c] > 0]
    for c in present:
        ap = summary['mean_dist_aps'][c]
        tp = summary['label_tp_errors'][c]
        if c not in with_gt:
            print('       %-14s no GT inside the class range in this split (devkit AP %.3f), not judged' % (c, ap))
            continue
        errs = [tp[k] for k in ('trans_err', 'scale_err', 'orient_err') if not np.isnan(tp[k])]
        report(abs(ap - 1.0) < 1e-6 and max(errs, default=0) < 1e-3,
               '%-14s AP %.4f, ATE %.1e m, ASE %.1e, AOE %.1e rad' % (c, ap, tp['trans_err'], tp['scale_err'],
                                                                   tp['orient_err']))
        # AAE is not 0 here: the attribute comes from default_attribute (velocity rule), not from the GT. The devkit's
        # AAE is a mean over recall 0.1..1, so it only roughly equals the plain mismatch rate; judge it loosely.
        if not np.isnan(tp['attr_err']) and c in attr_hits:
            right, n = attr_hits[c]
            expected = 1.0 - right / n
            report(abs(tp['attr_err'] - expected) < 0.1,
                   '%-14s AAE %.3f (devkit) vs %.3f attribute mismatch of default_attribute (%d boxes)'
                   % (c, tp['attr_err'], expected, n))


def check_training_items(args):
    print('[4] training items (augmentation on), per modality')
    for sensor in ('lidar', 'radar'):
        cfg, ds = build_dataset(args.cfg, args.dataroot, args.version, args.info_dir, training=True, sensor=sensor)
        print('       grid %s, voxel %s, range %s' % (ds.grid_size.tolist(), ds.voxel_size, ds.point_cloud_range.tolist()))
        rng = np.random.RandomState(0)
        times, npts, nbox = [], [], []
        for idx in rng.choice(len(ds), size=min(args.num_items, len(ds)), replace=False):
            t0 = time.time()
            item = ds[int(idx)]
            times.append(time.time() - t0)
            npts.append(len(item['points']))
            nbox.append(len(item['gt_boxes']))
            assert item['points'].shape[1] == 5 and item['gt_boxes'].shape[1] == 10, \
                (item['points'].shape, item['gt_boxes'].shape)
            pr = ds.point_cloud_range
            inside = ((item['points'][:, 0] >= pr[0]) & (item['points'][:, 0] <= pr[3]) &
                      (item['points'][:, 1] >= pr[1]) & (item['points'][:, 1] <= pr[4])).all()
            assert inside, 'points outside the x/y range after the crop'
        f = item['points'][:, 3]
        report(True, '%s: %d items, %.0f points, %.1f GT boxes each; load %.2f s/item (1 CPU core); last item feature '
               'range [%.2f, %.2f], time lag range [%.3f, %.3f] s'
               % (sensor, len(times), np.mean(npts), np.mean(nbox), np.mean(times), f.min(), f.max(),
                  item['points'][:, 4].min(), item['points'][:, 4].max()))


def check_vfe_keys():
    print('[5] DynamicMeanVFE: int64 voxel keys (TruckScenes grid, batch 16) and unchanged DSERT-RoLL voxels')
    import torch
    torch.Tensor.cuda = lambda self, *a, **k: self   # CPU run
    # The file alone (the backbone_3d package imports spconv, which a CPU-only machine may not have).
    spec = importlib.util.spec_from_file_location(
        'dynamic_mean_vfe', os.path.join(REPO, 'detection', 'al3d_det', 'models', 'modules', 'backbone_3d',
                                         'dynamic_mean_vfe.py'))
    vfe_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vfe_mod)
    DynamicMeanVFE = vfe_mod.DynamicMeanVFE

    def old_int32(points, pc_range, voxel, grid):
        pcr, vs, gs = torch.tensor(pc_range), torch.tensor(voxel), torch.tensor(grid)
        coords = torch.floor((points[:, 1:4] - pcr[0:3]) / vs).int()
        mask = ((coords >= 0) & (coords < gs)).all(1)
        points, coords = points[mask], coords[mask]
        sxyz, syz, sz = int(grid[0] * grid[1] * grid[2]), int(grid[1] * grid[2]), int(grid[2])
        key = points[:, 0].int() * sxyz + coords[:, 0] * syz + coords[:, 1] * sz + coords[:, 2]
        u = torch.unique(key).int()
        vc = torch.stack((torch.div(u, sxyz, rounding_mode='trunc'), torch.div(u % sxyz, syz, rounding_mode='trunc'),
                          torch.div(u % syz, sz, rounding_mode='trunc'), u % sz), 1)[:, [0, 3, 2, 1]]
        return vc

    gen = torch.Generator().manual_seed(0)
    for name, pc_range, voxel, bs in (('TruckScenes', [-150, -150, -5, 150, 150, 7], [0.1, 0.1, 0.2], 16),
                                      ('DSERT-RoLL', [0, -75.2, -2, 75.2, 75.2, 4], [0.1, 0.1, 0.15], 16)):
        pcr, vs = np.array(pc_range, dtype=np.float32), np.array(voxel, dtype=np.float32)
        grid = np.round((pcr[3:] - pcr[:3]) / vs).astype(np.int64)
        vfe = DynamicMeanVFE(None, 5, vs, grid, pcr)
        n = 20000
        b = torch.randint(0, bs, (n, 1), generator=gen).float()
        xyz = torch.tensor(pcr[:3]) + torch.rand(n, 3, generator=gen) * torch.tensor(pcr[3:] - pcr[:3])
        pts = torch.cat([b, xyz, torch.rand(n, 2, generator=gen)], 1)
        out = vfe({'batch_size': bs, 'points': pts.clone()})
        vc = out['voxel_coords'].long()   # (batch, z, y, x)
        exp = torch.floor((xyz - torch.tensor(pcr[:3])) / torch.tensor(vs)).long()
        exp_set = set(map(tuple, torch.cat([b.long(), exp[:, [2, 1, 0]]], 1).tolist()))
        got_set = set(map(tuple, vc.tolist()))
        report(got_set == exp_set, '%s grid %s, batch %d: %d voxels, all at their points\' cells'
               % (name, grid.tolist(), bs, len(got_set)))
        old = set(map(tuple, old_int32(pts, pc_range, voxel, grid).long().tolist()))
        if name == 'DSERT-RoLL':
            report(old == got_set, 'DSERT-RoLL: identical to the old int32 keys')
        else:
            print('       (old int32 keys on this grid: %d of %d voxels wrong)' % (len(old - exp_set), len(old)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataroot', required=True)
    ap.add_argument('--version', default='v1.2-mini')
    ap.add_argument('--info_dir', default=None)
    ap.add_argument('--cfg', default='cfgs/det_model_cfgs/truckscenes/voxelnext.yaml')
    ap.add_argument('--num_samples', type=int, default=20, help='samples for checks 1 and 2')
    ap.add_argument('--num_items', type=int, default=8, help='training items per modality for check 4')
    ap.add_argument('--out_dir', default=os.path.join(REPO, 'detection', 'output', 'truckscenes_data_check'))
    ap.add_argument('--skip', default='', help='comma-separated check numbers to skip, e.g. 3')
    args = ap.parse_args()
    args.cfg = os.path.join(HERE, args.cfg) if not os.path.isabs(args.cfg) else args.cfg
    skip = set(args.skip.split(',')) if args.skip else set()

    try:
        import al3d_utils.box_utils  # noqa: F401
    except ImportError:
        print('(compiled CUDA ops not found: importing stand-ins, fine for these CPU checks)')
        sys.meta_path.insert(0, _StubCompiledOps())
    from truckscenes import TruckScenes
    from al3d_det.datasets.man_truckscenes import truckscenes_utils as tu
    import pickle

    info_dir = args.info_dir or os.path.join(args.dataroot, 'al3d_infos', args.version)
    val_split = 'mini_val' if args.version.endswith('mini') else 'val'
    with open(os.path.join(info_dir, 'truckscenes_infos_%s.pkl' % val_split), 'rb') as f:
        infos = pickle.load(f)['infos']
    # spread the samples over the split (different scenes, and scene starts with fewer earlier sweeps)
    pick = np.linspace(0, len(infos) - 1, min(args.num_samples, len(infos))).astype(int)
    sel = [infos[i] for i in pick]
    print('%s %s: %d samples, checking %d' % (args.version, val_split, len(infos), len(sel)))

    trucksc = TruckScenes(version=args.version, dataroot=args.dataroot, verbose=False) if {'1'} - skip else None
    if '1' not in skip:
        check_points(trucksc, tu, sel, args.dataroot, len(sel))
    if '2' not in skip:
        check_boxes(tu, sel, args.dataroot, len(sel))
    if '3' not in skip:
        check_eval(args, os.path.join(args.out_dir, args.version))
    if '4' not in skip:
        check_training_items(args)
    if '5' not in skip:
        check_vfe_keys()

    print()
    if FAILURES:
        print('FAILED %d check(s):' % len(FAILURES))
        for f in FAILURES:
            print('  ' + f)
        sys.exit(1)
    print('ALL TRUCKSCENES DATA CHECKS PASSED')


if __name__ == '__main__':
    main()
