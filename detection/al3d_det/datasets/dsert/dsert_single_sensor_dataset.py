import os

import numpy as np

from .dsert_dataset import DSERTTrainingDataset


# Per sensor: its name in info['calibration'] (None = auto-detect), its key in info['sensor'] (None = auto-detect,
# falling back to swapping FOLDER into the Livox path), the per-point feature used as the 4th channel, and how
# that feature is rescaled.
SENSOR_DEFAULTS = {
    'livox': dict(CALIB_NAME='Livox', PATH_KEY='livox_path', FOLDER='LIDAR_LIVOX_Tilted',
                  FEATURE='intensity', INTENSITY_TRANSFORM='none'),
    'ouster': dict(CALIB_NAME=None, PATH_KEY=None, FOLDER='LIDAR_OUSTER_Tilted_90_degree',
                   FEATURE='intensity', INTENSITY_TRANSFORM='log1p'),
    'radar': dict(CALIB_NAME='Radar', PATH_KEY='radar_path', FOLDER='RADAR_Tilted',
                  FEATURE='doppler', INTENSITY_TRANSFORM='none'),
}
LIVOX_FOLDER = 'LIDAR_LIVOX_Tilted'
FEATURE_FALLBACKS = ['intensity', 'reflectivity', 'signal', 'doppler']
SENTINEL_POINT = np.array([[10.0, 0.0, 0.0, 0.0]], dtype=np.float32)


class DSERTSingleSensorDataset(DSERTTrainingDataset):
    """
    DSERT frames with points from a single sensor (DATA_CONFIG.SENSOR.NAME: livox | ouster | radar),
    transformed into the Livox frame, which is where the boxes are annotated.

    `points_radar` is always a single placeholder point. The augmentors expect the key to exist, and the
    single-sensor model never reads it, so nothing from the real radar reaches a LiDAR run.
    """

    def __init__(self, dataset_cfg, class_names, root_path, training=True, logger=None):
        sensor_cfg = dataset_cfg.get('SENSOR', {})
        self.sensor_name = str(sensor_cfg.get('NAME', 'livox')).lower()
        if self.sensor_name not in SENSOR_DEFAULTS:
            raise ValueError('SENSOR.NAME must be one of %s, got %s' % (list(SENSOR_DEFAULTS), self.sensor_name))
        self.sensor_opts = dict(SENSOR_DEFAULTS[self.sensor_name])
        # 'auto' (or null) keeps the per-sensor default, which for Ouster means auto-detect.
        self.sensor_opts.update({k: v for k, v in sensor_cfg.items() if k != 'NAME' and v not in (None, 'auto')})
        self._derive_path_from_livox = False
        self._feature_name = None
        super().__init__(dataset_cfg, class_names, root_path, training, logger)
        if len(self.infos) > 0:
            self._resolve_sensor(self.infos[0])

    def _resolve_sensor(self, info):
        """Pick the sensor's file key and calibration name once, and log the choice so it can be checked."""
        sensor = info['sensor']
        if self.sensor_opts.get('PATH_KEY') is None:
            candidates = [k for k in sensor if self.sensor_name in k.lower() and sensor[k]]
            if candidates:
                self.sensor_opts['PATH_KEY'] = candidates[0]
            elif LIVOX_FOLDER in str(sensor.get('livox_path', '')):
                self._derive_path_from_livox = True
            else:
                raise KeyError('No %s path in info["sensor"] (keys: %s). Set DATA_CONFIG.SENSOR.PATH_KEY.'
                               % (self.sensor_name, list(sensor)))
        elif self.sensor_opts['PATH_KEY'] not in sensor:
            raise KeyError('SENSOR.PATH_KEY=%s not in info["sensor"] (keys: %s)'
                           % (self.sensor_opts['PATH_KEY'], list(sensor)))

        calib = info['calibration']
        if self.sensor_opts.get('CALIB_NAME') is None:
            candidates = [k for k in calib if self.sensor_name in str(k).lower()]
            if not candidates:
                raise KeyError('No %s entry in info["calibration"] (keys: %s). Set DATA_CONFIG.SENSOR.CALIB_NAME.'
                               % (self.sensor_name, list(calib)))
            self.sensor_opts['CALIB_NAME'] = candidates[0]
        T = self.sensor_to_livox(info)

        # Load the first frame here, in the main process: it fails fast on a wrong path and fixes the feature
        # field before the dataset is pickled into dataloader workers (which drop the logger).
        first_path = os.path.join(self.data_path, self.sensor_rel_path(info))
        if not os.path.exists(first_path):
            raise FileNotFoundError('%s frame not found: %s' % (self.sensor_name, first_path))
        raw = np.load(first_path)
        names = raw.dtype.names
        if names is not None:
            wanted = [self.sensor_opts['FEATURE']] + FEATURE_FALLBACKS
            self._feature_name = next((f for f in wanted if f in names), '')

        if self.logger is not None:
            if names is not None and self._feature_name != self.sensor_opts['FEATURE']:
                self.logger.warning('%s points have fields %s; using "%s" as the 4th channel'
                                    % (self.sensor_name, names, self._feature_name or 'zeros'))
            path_desc = ('derived from livox_path by swapping in %s' % self.sensor_opts['FOLDER']
                         if self._derive_path_from_livox else 'info["sensor"]["%s"]' % self.sensor_opts['PATH_KEY'])
            self.logger.info('Single-sensor dataset: sensor=%s, path=%s, calib=%s, intensity transform=%s'
                             % (self.sensor_name, path_desc, self.sensor_opts['CALIB_NAME'],
                                self.sensor_opts['INTENSITY_TRANSFORM']))
            self.logger.info('%s -> Livox transform (first frame):\n%s' % (self.sensor_name, np.array2string(T, precision=4)))

    def sensor_rel_path(self, info):
        if self._derive_path_from_livox:
            return info['sensor']['livox_path'].replace(LIVOX_FOLDER, self.sensor_opts['FOLDER'])
        return info['sensor'][self.sensor_opts['PATH_KEY']]

    def sensor_to_livox(self, info):
        """4x4 transform from the sensor frame to the Livox frame. calibration[A][B] is the A -> B transform."""
        calib = info['calibration']
        name = self.sensor_opts['CALIB_NAME']
        if name == 'Livox':
            return np.eye(4)
        src = calib.get(name, {})
        livox = calib.get('Livox', {})
        src = src if isinstance(src, dict) else {}
        livox = livox if isinstance(livox, dict) else {}
        if 'Livox' in src:
            return self._as_T(src['Livox'])
        if name in livox:
            return self.inverse_T(self._as_T(livox[name]))
        # Chain through a frame both sensors are calibrated to, the same way the fusion dataset maps radar.
        common = [k for k in src if k in livox]
        if not common:
            raise KeyError('Cannot build %s -> Livox: calibration["%s"] has %s, calibration["Livox"] has %s'
                           % (name, name, list(src), list(livox)))
        via = 'Thermal_L' if 'Thermal_L' in common else common[0]
        return self.inverse_T(self._as_T(livox[via])) @ self._as_T(src[via])

    @staticmethod
    def _as_T(m):
        T = np.asarray(m, dtype=np.float64)
        assert T.shape == (4, 4), 'expected a 4x4 transform, got shape %s' % (T.shape,)
        return T

    def load_sensor_points(self, path):
        """Returns (N, 4) float32 points [x, y, z, feature] in the sensor's own frame."""
        raw = np.load(path)
        if raw.dtype.names is not None:
            if self._feature_name is None:
                wanted = [self.sensor_opts['FEATURE']] + FEATURE_FALLBACKS
                self._feature_name = next((f for f in wanted if f in raw.dtype.names), '')
            feat = raw[self._feature_name] if self._feature_name else np.zeros(raw.shape[0])
            pts = np.stack([raw['x'], raw['y'], raw['z'], feat], -1)
        else:
            arr = raw.reshape(-1, raw.shape[-1])
            feat = arr[:, 3] if arr.shape[1] > 3 else np.zeros(arr.shape[0])
            pts = np.concatenate([arr[:, :3], feat[:, None]], -1)
        pts = pts.astype(np.float32)
        # Organized LiDAR scans (Ouster) store missing returns as zeros or NaNs.
        keep = np.isfinite(pts).all(1) & (np.abs(pts[:, :3]).sum(1) > 0)
        return pts[keep]

    def _transform_feature(self, feat):
        mode = self.sensor_opts['INTENSITY_TRANSFORM']
        if mode == 'none':
            return feat
        if mode == 'log1p':
            # Ouster intensity reaches tens of thousands, past what fp16 activations handle well.
            return np.log1p(np.maximum(feat, 0.0))
        raise ValueError('Unknown INTENSITY_TRANSFORM %s' % mode)

    def get_infos_and_points(self, idx_list):
        infos, points, points_radar = [], [], []
        for i in idx_list:
            info = self.infos[i]
            pts = self.load_sensor_points(os.path.join(self.data_path, self.sensor_rel_path(info)))
            T = self.sensor_to_livox(info)
            pts[:, :3] = (pts[:, :3].astype(np.float64) @ T[:3, :3].T + T[:3, 3]).astype(np.float32)
            pts[:, 3] = self._transform_feature(pts[:, 3])

            # Same sentinel as the fusion dataset, so downstream tensors are never empty.
            infos.append(info)
            points.append(np.concatenate([pts, SENTINEL_POINT], 0))
            points_radar.append(SENTINEL_POINT.copy())
        return infos, points, points_radar
