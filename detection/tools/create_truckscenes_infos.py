"""
Build the info files TruckScenesDataset trains and evaluates from, using only the official devkit
(truckscenes-devkit submodule): the official split (truckscenes.utils.splits), GT boxes and velocities
(TruckScenes.get_box / box_velocity / boxes_to_sensor), the detection-class mapping (category_to_detection_name) and,
for every LiDAR and radar channel, the sweep files plus the devkit's sensor -> reference-ego transforms
(see al3d_det/datasets/man_truckscenes/truckscenes_utils.py). No point cloud is read or rewritten here; the dataset reads
the original .pcd files with the devkit's LidarPointCloud / RadarPointCloud at training time.

    python create_truckscenes_infos.py --dataroot /path/to/man-truckscenes --version v1.2-trainval
    python create_truckscenes_infos.py --dataroot /path/to/man-truckscenes --version v1.2-mini

Writes <dataroot>/al3d_infos/<version>/truckscenes_infos_<split>.pkl for the version's two splits
(train/val, or mini_train/mini_val). Each file holds {'infos': [...], 'meta': {...}}; meta['attr_dist'] counts the GT
attributes per detection class in that split (used for the default attribute of a prediction, as OpenPCDet does).

--max_lidar_sweeps / --max_radar_sweeps: how many sweeps per channel to list (keyframe + earlier ones). The defaults
are the most the TruckScenes detection rules allow at inference (6 past LiDAR sweeps, 11 past radar sweeps); the
dataset config picks how many it actually uses (DATA_CONFIG.SENSOR.NSWEEPS).
"""
import argparse
import json
import os
import pickle
import sys
import time
from collections import Counter, defaultdict

import importlib.util

from pyquaternion import Quaternion


def load_truckscenes_utils():
    """al3d_det/datasets/man_truckscenes/truckscenes_utils.py, loaded by path: it needs only the devkit, so this script
    runs without torch, spconv or the repo's compiled ops (e.g. on a login node)."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'al3d_det', 'datasets', 'man_truckscenes',
                        'truckscenes_utils.py')
    spec = importlib.util.spec_from_file_location('truckscenes_utils', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


tu = load_truckscenes_utils()


def split_names(version):
    if version.endswith('mini'):
        return ['mini_train', 'mini_val']
    if version.endswith('trainval'):
        return ['train', 'val']
    raise ValueError('use a trainval or mini version (test has no annotations): %s' % version)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataroot', required=True)
    ap.add_argument('--version', default='v1.2-trainval')
    ap.add_argument('--out_dir', default=None, help='default: <dataroot>/al3d_infos/<version>')
    ap.add_argument('--max_lidar_sweeps', type=int, default=7)
    ap.add_argument('--max_radar_sweeps', type=int, default=12)
    args = ap.parse_args()

    from truckscenes import TruckScenes
    from truckscenes.utils.splits import create_splits_scenes

    out_dir = args.out_dir or os.path.join(args.dataroot, 'al3d_infos', args.version)
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    trucksc = TruckScenes(version=args.version, dataroot=args.dataroot, verbose=True)
    print('loaded %s in %.0fs' % (args.version, time.time() - t0))
    lidar_chans, radar_chans = tu.channels(trucksc, 'lidar'), tu.channels(trucksc, 'radar')
    print('lidar channels:', lidar_chans)
    print('radar channels:', radar_chans)

    scene_splits = create_splits_scenes()
    scenes_by_name = {s['name']: s for s in trucksc.scene}
    for split in split_names(args.version):
        names = [n for n in scene_splits[split] if n in scenes_by_name]
        missing = len(scene_splits[split]) - len(names)
        if missing:
            print('WARNING: %d scenes of split %s are not in %s' % (missing, split, args.version))
        infos = []
        attr_dist = defaultdict(Counter)
        t0 = time.time()
        for scene_name in names:
            scene = scenes_by_name[scene_name]
            sample_tokens = []
            tok = scene['first_sample_token']
            while tok:
                sample_tokens.append(tok)
                tok = trucksc.get('sample', tok)['next']
            for idx, tok in enumerate(sample_tokens):
                sample = trucksc.get('sample', tok)
                ref_sd = trucksc.get('sample_data', sample['data'][tu.REF_CHANNEL])
                ref_pose = trucksc.get('ego_pose', ref_sd['ego_pose_token'])
                annos = tu.sample_boxes_ego(trucksc, sample)
                for n, a in zip(annos['name'], annos['attribute_name']):
                    if n:
                        attr_dist[n][a] += 1
                infos.append({
                    'sample_token': tok,
                    'scene_token': scene['token'],
                    'sequence_name': scene_name,
                    'sample_idx': idx,
                    'sequence_len': len(sample_tokens),
                    'time_stamp': int(ref_sd['timestamp']),
                    'ego_translation': list(ref_pose['translation']),
                    'ego_rotation': list(ref_pose['rotation']),
                    'pose': tu.transform_matrix(ref_pose['translation'], Quaternion(ref_pose['rotation'])),
                    'sweeps': {
                        'lidar': {c: tu.sweep_chain(trucksc, sample, c, args.max_lidar_sweeps) for c in lidar_chans},
                        'radar': {c: tu.sweep_chain(trucksc, sample, c, args.max_radar_sweeps) for c in radar_chans},
                    },
                    'annos': annos,
                })
        meta = {'version': args.version, 'split': split, 'ref_channel': tu.REF_CHANNEL,
                'lidar_channels': lidar_chans, 'radar_channels': radar_chans,
                'max_sweeps': {'lidar': args.max_lidar_sweeps, 'radar': args.max_radar_sweeps},
                'attr_dist': {k: dict(v) for k, v in attr_dist.items()}}
        path = os.path.join(out_dir, 'truckscenes_infos_%s.pkl' % split)
        with open(path, 'wb') as f:
            pickle.dump({'infos': infos, 'meta': meta}, f, protocol=4)
        with open(os.path.join(out_dir, 'truckscenes_attr_dist_%s.json' % split), 'w') as f:
            json.dump(meta['attr_dist'], f, indent=1)
        n_boxes = sum(int((i['annos']['name'] != '').sum()) for i in infos)
        print('%s: %d scenes, %d samples, %d boxes of detection classes -> %s (%.0fs)'
              % (split, len(names), len(infos), n_boxes, path, time.time() - t0))


if __name__ == '__main__':
    main()
