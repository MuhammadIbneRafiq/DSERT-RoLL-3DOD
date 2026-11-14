"""
MAN TruckScenes for the single-sensor detectors (VoxelNeXtSingle), with the official devkit doing all data access and
the scoring:

- infos from tools/create_truckscenes_infos.py (official split, GT from the devkit);
- points: all channels of one modality (DATA_CONFIG.SENSOR.NAME: lidar = the 6 LiDARs, radar = the 6 radars), each
  with its first SENSOR.NSWEEPS sweeps, merged in the ego frame of the LIDAR_LEFT keyframe with the devkit's
  calibrated_sensor + ego_pose transforms (truckscenes_utils.load_sweeps). Per point: x, y, z, feature, time_lag;
- evaluation: predictions are written as a TruckScenes submission (global frame) and scored by the devkit's own
  truckscenes.eval.detection.evaluate.DetectionEval with config detection_cvpr_2024 (mAP over 0.5/1/2/4 m centre
  distance, TP errors, NDS; class ranges 150 m / 75 m; GT boxes without LiDAR or radar points removed).

`points_radar` is a single placeholder point, as in DSERTSingleSensorDataset: the augmentors expect the key, the
single-sensor model never reads it.
"""
import copy
import json
import os
import pickle

import numpy as np

from al3d_det.datasets.dataset import DatasetTemplate
from . import truckscenes_utils as tu

# Defaults per modality: which per-point value is the 4th channel.
FEATURE_DEFAULTS = {'lidar': 'intensity', 'radar': 'doppler'}
VALID_FEATURES = {'lidar': ('intensity', 'none'), 'radar': ('doppler', 'rcs', 'none')}

# Classes without attributes in TruckScenes (truckscenes.eval.detection.utils.detection_name_to_rel_attributes).
NO_ATTRIBUTE = ('barrier', 'traffic_cone', 'animal')


class TruckScenesDataset(DatasetTemplate):

    def __init__(self, dataset_cfg, class_names, root_path=None, training=True, logger=None):
        super().__init__(dataset_cfg, class_names, training, root_path, logger)
        sensor_cfg = dataset_cfg.get('SENSOR', {})
        self.modality = str(sensor_cfg.get('NAME', 'lidar')).lower()
        assert self.modality in FEATURE_DEFAULTS, 'SENSOR.NAME must be lidar or radar, got %s' % self.modality
        feature = sensor_cfg.get('FEATURE', 'auto')
        self.feature = FEATURE_DEFAULTS[self.modality] if feature in (None, 'auto') else str(feature).lower()
        assert self.feature in VALID_FEATURES[self.modality], \
            'SENSOR.FEATURE for %s must be one of %s' % (self.modality, VALID_FEATURES[self.modality])
        self.nsweeps = int(sensor_cfg.NSWEEPS[self.modality] if isinstance(sensor_cfg.get('NSWEEPS'), dict)
                           else sensor_cfg.get('NSWEEPS', 1))
        self.channels = sensor_cfg.get('CHANNELS', 'all')
        self.version = dataset_cfg.VERSION
        self.info_dir = dataset_cfg.get('INFO_DIR', None) or os.path.join(self.root_path, 'al3d_infos', self.version)
        self.split = dataset_cfg.DATA_SPLIT[self.mode]
        self.init_infos()

    # ------------------------------------------------------------------ infos
    def _info_path(self, split):
        return os.path.join(self.info_dir, 'truckscenes_infos_%s.pkl' % split)

    def init_infos(self):
        with open(self._info_path(self.split), 'rb') as f:
            data = pickle.load(f)
        meta = data['meta']
        assert meta['version'] == self.version, 'info file is for %s, config says %s' % (meta['version'], self.version)
        assert self.nsweeps <= meta['max_sweeps'][self.modality], \
            'NSWEEPS %d > the %d sweeps listed in the info file' % (self.nsweeps, meta['max_sweeps'][self.modality])
        all_chans = meta['%s_channels' % self.modality]
        chans = all_chans if self.channels in (None, 'all') else list(self.channels)
        assert set(chans) <= set(all_chans), 'unknown channels %s (have %s)' % (chans, all_chans)
        self.channels = chans

        infos = data['infos']
        self.split_sample_tokens = [i['sample_token'] for i in infos]   # the whole split, for the evaluation
        # Keep only what this run reads (the other modality and unused sweeps), so dataloader workers stay small.
        for info in infos:
            info['sweeps'] = {c: info['sweeps'][self.modality][c][:self.nsweeps] for c in self.channels}
        interval = self.dataset_cfg.SAMPLED_INTERVAL[self.mode]
        self.infos = infos[::interval] if interval > 1 else infos

        attr_path = os.path.join(self.info_dir, 'truckscenes_attr_dist_%s.json' % self.dataset_cfg.DATA_SPLIT['train'])
        self.attr_dist = json.load(open(attr_path)) if os.path.exists(attr_path) else {}
        if self.logger is not None:
            self.logger.info('TruckScenes %s / %s: %d samples (of %d), %s: %s, %d sweep(s) each, 4th channel %s'
                             % (self.version, self.split, len(self.infos), len(infos), self.modality,
                                ','.join(self.channels), self.nsweeps, self.feature))
            if not self.attr_dist:
                self.logger.warning('no %s: predicted boxes get no attribute (affects mAAE/NDS only)' % attr_path)

    # ------------------------------------------------------------------ points + boxes
    def load_points(self, info):
        return tu.load_sweeps(self.root_path, info['sweeps'], self.modality, self.nsweeps, info['time_stamp'],
                              self.feature)

    def __getitem__(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.infos)
        info = self.infos[index]
        points = np.concatenate([self.load_points(info), tu.SENTINEL_POINT], 0)
        input_dict = {
            'points': points,
            'points_radar': tu.SENTINEL_POINT.copy(),
            'frame_id': info['sample_token'],
            'pose': info['pose'],
            'sequence_name': info['sequence_name'],
        }
        annos = info['annos']
        keep = annos['name'] != ''
        if self.dataset_cfg.get('FILTER_EMPTY_GT', True):
            # As the official evaluation: GT boxes without any LiDAR or radar point are not counted.
            keep &= (annos['num_lidar_pts'] + annos['num_radar_pts']) > 0
        gt_boxes = annos['gt_boxes_ego'][keep].astype(np.float32)
        if self.dataset_cfg.get('SET_NAN_VELOCITY_TO_ZEROS', True):
            gt_boxes[np.isnan(gt_boxes)] = 0.0
        input_dict['gt_names'] = annos['name'][keep]
        input_dict['gt_boxes'] = gt_boxes
        return self.prepare_data(data_dict=input_dict)

    def generate_prediction_dicts(self, batch_dict, pred_dicts, class_names, output_path=None):
        # frame_id is the sample token (a string), so the per-frame pickles of the base class are not written.
        return DatasetTemplate.generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path=None)

    # ------------------------------------------------------------------ evaluation
    def default_attribute(self, name, velocity):
        """OpenPCDet's rule for nuScenes (nuscenes_utils.transform_det_annos_to_nusc_annos), with TruckScenes' class
        names and the attribute frequencies of the TruckScenes train split. Affects mAAE / NDS only, not mAP."""
        if name in NO_ATTRIBUTE:
            return ''
        attr = None
        if np.hypot(velocity[0], velocity[1]) > 0.2:
            if name in ('car', 'truck', 'bus', 'trailer', 'other_vehicle'):
                attr = 'vehicle.moving'
            elif name in ('bicycle', 'motorcycle'):
                attr = 'cycle.with_rider'
        else:
            if name == 'pedestrian':
                attr = 'pedestrian.standing'
            elif name == 'bus':
                attr = 'vehicle.stopped'
        if attr is None:
            dist = {k: v for k, v in self.attr_dist.get(name, {}).items() if k}
            attr = max(dist.items(), key=lambda kv: kv[1])[0] if dist else ''
        return attr

    def to_submission(self, det_annos):
        info_by_token = {i['sample_token']: i for i in self.infos}
        results = {}
        for det in det_annos:
            token = str(det['frame_id'])
            info = info_by_token[token]
            boxes = []
            for k in range(len(det['name'])):
                box = tu.ego_box_to_global(det['boxes_lidar'][k], info['ego_translation'], info['ego_rotation'])
                name = str(det['name'][k])
                boxes.append({
                    'sample_token': token,
                    'translation': box.center.tolist(),
                    'size': box.wlh.tolist(),
                    'rotation': box.orientation.elements.tolist(),
                    'velocity': box.velocity[:2].tolist(),
                    'detection_name': name,
                    'detection_score': float(det['score'][k]),
                    'attribute_name': self.default_attribute(name, box.velocity),
                })
            results[token] = boxes
        return results

    def evaluation(self, det_annos, class_names, **kwargs):
        from truckscenes import TruckScenes
        from truckscenes.eval.detection.config import config_factory
        from truckscenes.eval.detection.evaluate import DetectionEval

        output_dir = str(kwargs.get('output_path') or os.path.join(self.root_path, 'eval_tmp'))
        os.makedirs(output_dir, exist_ok=True)
        results = self.to_submission(det_annos)
        missing = [t for t in self.split_sample_tokens if t not in results]
        official = not missing
        if missing:
            # The devkit evaluates whole splits only. A subsampled run (SMOKE) gets empty predictions for the samples it
            # skipped, so the devkit runs unchanged, but the numbers are NOT official (recall is capped).
            for t in missing:
                results[t] = []
        submission = {
            'meta': {'use_camera': False, 'use_lidar': self.modality == 'lidar', 'use_radar': self.modality == 'radar',
                     'use_map': False, 'use_external': False, 'use_future_frames': False, 'use_tta': False,
                     'method_name': 'VoxelNeXt (%s, %d sweeps)' % (self.modality, self.nsweeps), 'authors': '',
                     'affiliation': '', 'description': '', 'code_url': '', 'paper_url': ''},
            'results': results,
        }
        result_path = os.path.join(output_dir, 'results_truckscenes.json')
        with open(result_path, 'w') as f:
            json.dump(submission, f)

        eval_cfg = self.dataset_cfg.get('EVAL', {})
        trucksc = TruckScenes(version=self.version, dataroot=self.root_path, verbose=False)
        evaluator = DetectionEval(trucksc, config=config_factory(eval_cfg.get('CONFIG', 'detection_cvpr_2024')),
                                  result_path=result_path, eval_set=self.split,
                                  output_dir=os.path.join(output_dir, 'truckscenes_eval'), verbose=True)
        summary = evaluator.main(plot_examples=0, render_curves=bool(eval_cfg.get('RENDER_CURVES', False)),
                                 evaluate_tags=bool(eval_cfg.get('EVALUATE_TAGS', False)))
        allm = summary['all']

        lines = ['==== TruckScenes official detection evaluation (devkit DetectionEval, %s, split %s) ===='
                 % (eval_cfg.get('CONFIG', 'detection_cvpr_2024'), self.split)]
        if not official:
            lines.append('WARNING: predictions for %d of %d samples only (subsampled run): NOT an official number'
                         % (len(self.split_sample_tokens) - len(missing), len(self.split_sample_tokens)))
        lines.append('mAP: %.4f   NDS: %.4f' % (allm['mean_ap'], allm['nd_score']))
        lines.append('%-14s %6s' % ('class', 'AP'))
        ret = {'mAP': allm['mean_ap'], 'NDS': allm['nd_score'], 'official': float(official)}
        for name, ap in allm['mean_dist_aps'].items():
            lines.append('%-14s %6.4f' % (name, ap))
            ret['AP/%s' % name] = ap
        for name, err in allm['tp_errors'].items():
            ret['TP/%s' % name] = err
        lines.append('submission: %s' % result_path)
        return '\n'.join(lines) + '\n', ret
