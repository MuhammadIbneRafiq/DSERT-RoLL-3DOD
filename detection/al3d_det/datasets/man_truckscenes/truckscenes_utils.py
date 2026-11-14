"""
MAN TruckScenes helpers, kept as close to the official devkit (truckscenes-devkit submodule, `truckscenes` package) as
possible. Everything that touches the data goes through devkit code:

- point clouds: truckscenes.utils.data_classes.LidarPointCloud / RadarPointCloud (from_file, remove_close, transform);
- sweep accumulation: the same chain of transforms as PointCloud.from_file_multisweep, except that the result is
  left in the ego-vehicle frame of the reference keyframe instead of being moved on into the reference sensor's frame
  (devkit: ref_from_car @ car_from_global @ global_from_car @ car_from_current; here: car_from_global @
  global_from_car @ car_from_current). check_truckscenes_data.py checks this against from_file_multisweep itself;
- boxes: TruckScenes.get_box / box_velocity / boxes_to_sensor, and the detection-class mapping
  truckscenes.eval.detection.utils.category_to_detection_name;
- splits: truckscenes.utils.splits.create_splits_scenes.

The ego-vehicle frame is the devkit's: origin at the rear axle centre on the ground, x forward, y left, z up (ISO 8855).
The reference keyframe is LIDAR_LEFT, the channel whose ego pose the official evaluation uses for box distances
(truckscenes.eval.common.loaders.add_center_dist).
"""
import os
from functools import reduce

import numpy as np
from pyquaternion import Quaternion

REF_CHANNEL = 'LIDAR_LEFT'
MIN_DISTANCE = 1.0     # devkit default of from_file_multisweep: drop points within 1 m (x and y) of the sensor
SENTINEL_POINT = np.array([[10.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
IDENTITY_CS = {'translation': [0.0, 0.0, 0.0], 'rotation': [1.0, 0.0, 0.0, 0.0]}


def transform_matrix(translation, rotation, inverse=False):
    from truckscenes.utils.geometry_utils import transform_matrix as _tm
    return _tm(translation, rotation, inverse=inverse)


def channels(trucksc, modality):
    """All sensor channels of one modality ('lidar' | 'radar'), from the devkit's sensor table."""
    return sorted(s['channel'] for s in trucksc.sensor if s['modality'] == modality)


def sweep_chain(trucksc, sample_rec, chan, nsweeps, ref_chan=REF_CHANNEL):
    """
    The keyframe of `chan` in this sample and up to nsweeps - 1 earlier sweeps, each with the 4x4 transform from that
    sweep's sensor frame to the ego frame of the reference keyframe. Same walk and same transforms as the devkit's
    PointCloud.from_file_multisweep (it stops early at the start of a scene).
    """
    ref_sd = trucksc.get('sample_data', sample_rec['data'][ref_chan])
    ref_pose = trucksc.get('ego_pose', ref_sd['ego_pose_token'])
    car_from_global = transform_matrix(ref_pose['translation'], Quaternion(ref_pose['rotation']), inverse=True)

    out = []
    sd = trucksc.get('sample_data', sample_rec['data'][chan])
    for _ in range(nsweeps):
        pose = trucksc.get('ego_pose', sd['ego_pose_token'])
        global_from_car = transform_matrix(pose['translation'], Quaternion(pose['rotation']), inverse=False)
        cs = trucksc.get('calibrated_sensor', sd['calibrated_sensor_token'])
        car_from_current = transform_matrix(cs['translation'], Quaternion(cs['rotation']), inverse=False)
        T = reduce(np.dot, [car_from_global, global_from_car, car_from_current])
        out.append({'filename': sd['filename'], 'T': T.astype(np.float64), 'timestamp': int(sd['timestamp'])})
        if sd['prev'] == '':
            break
        sd = trucksc.get('sample_data', sd['prev'])
    return out


def load_sweeps(dataroot, sweeps_by_channel, modality, nsweeps, ref_time_us, feature):
    """
    Points of all channels and their first nsweeps sweeps, in the reference ego frame, as (N, 5) float32
    [x, y, z, feature, time_lag]. time_lag (s) is computed as in the devkit's from_file_multisweep: per point for
    LiDAR (each point has its own timestamp), per sweep for radar.

    feature: lidar 'intensity' | 'none'; radar 'doppler' | 'rcs' | 'none'.
      doppler = the signed radial relative velocity, vrel . p / |p| in the sensor frame (the radar's vrel_x/y/z are
      the relative radial velocity resolved into x/y/z, not ego-motion compensated). A scalar, so it is unaffected by
      the flip/rotation augmentations, as DSERT-RoLL's radar Doppler channel.
    """
    from truckscenes.utils.data_classes import LidarPointCloud, RadarPointCloud
    cls = LidarPointCloud if modality == 'lidar' else RadarPointCloud
    ref_time = 1e-6 * ref_time_us
    parts = []
    for chan in sorted(sweeps_by_channel):
        for sw in sweeps_by_channel[chan][:nsweeps]:
            pc = cls.from_file(os.path.join(dataroot, sw['filename']))
            pc.remove_close(MIN_DISTANCE)
            n = pc.nbr_points()
            if n == 0:
                continue
            if modality == 'lidar':
                feat = pc.points[3] if feature == 'intensity' else np.zeros(n)
            elif feature == 'doppler':
                xyz, vrel = pc.points[:3], pc.points[3:6]
                feat = (xyz * vrel).sum(0) / np.maximum(np.linalg.norm(xyz, axis=0), 1e-6)
            elif feature == 'rcs':
                feat = pc.points[6]
            else:
                feat = np.zeros(n)
            if pc.timestamps is not None:
                time_lag = ref_time - 1e-6 * pc.timestamps.reshape(-1).astype(np.float64)
            else:
                time_lag = np.full(n, ref_time - 1e-6 * sw['timestamp'])
            pc.transform(sw['T'])
            parts.append(np.stack([pc.points[0], pc.points[1], pc.points[2], feat, time_lag], 1))
    if not parts:
        return np.zeros((0, 5), dtype=np.float32)
    return np.concatenate(parts, 0).astype(np.float32)


def sample_boxes_ego(trucksc, sample_rec, ref_chan=REF_CHANNEL):
    """
    GT boxes of a sample in the reference ego frame: (N, 9) [x, y, z, dx (length), dy (width), dz (height), yaw,
    vx, vy] plus per-box detection name ('' = not a detection class), attribute name, lidar/radar point counts and
    annotation tokens. Velocities are the devkit's box_velocity (NaN when it cannot be estimated).
    """
    from truckscenes.eval.detection.utils import category_to_detection_name
    ref_sd = trucksc.get('sample_data', sample_rec['data'][ref_chan])
    ref_pose = trucksc.get('ego_pose', ref_sd['ego_pose_token'])

    boxes, names, attrs, n_lidar, n_radar, tokens = [], [], [], [], [], []
    for ann_token in sample_rec['anns']:
        ann = trucksc.get('sample_annotation', ann_token)
        box = trucksc.get_box(ann_token)
        box.velocity = trucksc.box_velocity(ann_token)
        box = trucksc.boxes_to_sensor([box], ref_pose, IDENTITY_CS)[0]   # global -> ego (identity sensor)
        w, l, h = box.wlh
        boxes.append([*box.center, l, w, h, box.orientation.yaw_pitch_roll[0], box.velocity[0], box.velocity[1]])
        names.append(category_to_detection_name(ann['category_name']) or '')
        attrs.append(trucksc.get('attribute', ann['attribute_tokens'][0])['name'] if ann['attribute_tokens'] else '')
        n_lidar.append(ann['num_lidar_pts'])
        n_radar.append(ann['num_radar_pts'])
        tokens.append(ann_token)
    return {
        'gt_boxes_ego': np.array(boxes, dtype=np.float64).reshape(-1, 9),
        'name': np.array(names, dtype='<U16'),
        'attribute_name': np.array(attrs, dtype='<U32'),
        'num_lidar_pts': np.array(n_lidar, dtype=np.int64),
        'num_radar_pts': np.array(n_radar, dtype=np.int64),
        'token': np.array(tokens),
    }


def ego_box_to_global(box9, pose_translation, pose_rotation):
    """One predicted box (x, y, z, dx, dy, dz, yaw, vx, vy) in the reference ego frame -> devkit Box in global frame."""
    from truckscenes.utils.data_classes import Box
    x, y, z, dx, dy, dz, yaw = [float(v) for v in box9[:7]]
    vel = (float(box9[7]), float(box9[8]), 0.0) if len(box9) > 8 else (0.0, 0.0, 0.0)
    box = Box([x, y, z], [dy, dx, dz], Quaternion(axis=[0, 0, 1], radians=yaw), velocity=vel)
    # Inverse of boxes_to_sensor's ego step: rotate by the ego rotation, then translate by the ego position.
    box.rotate(Quaternion(pose_rotation))
    box.translate(np.array(pose_translation))
    return box
