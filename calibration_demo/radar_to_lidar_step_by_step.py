"""
radar_to_lidar_step_by_step.py: follow ONE real DSERT-RoLL frame through the radar -> LiDAR conversion, printing every
number, and draw what it looks like before and after. Run from anywhere:

    python calibration_demo/radar_to_lidar_step_by_step.py                       # Fog sample, frame 10
    python calibration_demo/radar_to_lidar_step_by_step.py --weather Light_Snow --frame 40

The conversion it reproduces is in the training code here:
    detection/al3d_det/datasets/dsert/dsert_dataset.py:164-169                (fusion model)
        radar2thermal = self.infos[i]['calibration']['Radar']['Thermal_L']
        thermal2lidar = self.inverse_T(self.infos[i]['calibration']['Livox']['Thermal_L'])
        radar2lidar = thermal2lidar @ radar2thermal
        new_radar_points = (radar2lidar @ np.stack([r_load['x'], r_load['y'], r_load['z'], ones], -1).T).T
    detection/al3d_det/datasets/dsert/dsert_single_sensor_dataset.py:119, 162-163   (your lidar_only radar runs)
        return self.inverse_T(self._as_T(livox[via])) @ self._as_T(src[via])         # via = 'Thermal_L'
        pts[:, :3] = (pts[:, :3] @ T[:3, :3].T + T[:3, 3])
Both compute the same matrix; this script checks that too.
"""
import argparse
import glob
import os
import pickle

import numpy as np

DEFAULT_ROOT = r"C:\Users\wifi stuff\dsert-roll-sample\processed_data"
np.set_printoptions(precision=3, suppress=True)


def step(n, title):
    print("\n" + "-" * 100 + "\nSTEP %s  %s\n" % (n, title) + "-" * 100)


def inverse_T(T):  # same as DSERTTrainingDataset.inverse_T (dsert_dataset.py:192)
    R_inv = np.linalg.inv(T[:3, :3])
    T_inv = np.eye(4)
    T_inv[:3, :3] = R_inv
    T_inv[:3, 3] = -R_inv @ T[:3, 3]
    return T_inv


def azimuth(p):
    """Horizontal angle of a point seen from the ORIGIN of its frame, measured from that frame's forward (x) axis,
    + = to the left. Only meaningful for frames with x forward, y left (Livox, radar)."""
    return np.degrees(np.arctan2(p[1], p[0]))


def in_box(p, b):
    d = p[:, :3] - b[:3]
    c, s = np.cos(-b[6]), np.sin(-b[6])
    lx, ly = d[:, 0] * c - d[:, 1] * s, d[:, 0] * s + d[:, 1] * c
    return (np.abs(lx) <= b[3] / 2 + .25) & (np.abs(ly) <= b[4] / 2 + .25) & (np.abs(d[:, 2]) <= b[5] / 2 + .5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT, help="processed_data folder (holds <Weather>/<drive>/label.pkl)")
    ap.add_argument("--weather", default="Fog")
    ap.add_argument("--frame", type=int, default=10, help="index into the drive's frame list")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "output"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # ------------------------------------------------------------------------------------------------------------
    step(1, "Open label.pkl: one per drive; calibration is stored once (meta), boxes and file paths per frame (info)")
    label = sorted(glob.glob(os.path.join(args.root, args.weather, "*", "label.pkl")))[0]
    z = pickle.load(open(label, "rb"))
    meta, info = z["meta"], z["info"][args.frame]
    cal = meta["calibration"]
    print("file:", label)
    print("top-level keys:", list(z))
    print("meta keys:     ", list(meta), "  weather =", meta.get("weather"), " light =", meta.get("light"))
    print("frames in drive:", len(z["info"]))
    print("info keys:     ", list(info))
    print("sensor files of this frame (relative to processed_data):")
    for k, v in info["sensor"].items():
        print("   %-22s %s" % (k, v))
    print("calibration has an entry per sensor:")
    for a, row in cal.items():
        print("   %-10s -> %s" % (a, list(row) if isinstance(row, dict) else type(row).__name__))

    # ------------------------------------------------------------------------------------------------------------
    step(2, "The two matrices the code uses. calibration[A][B] moves a point from A's frame into B's frame")
    radar2thermal = np.asarray(cal["Radar"]["Thermal_L"], np.float64)
    lidar2thermal = np.asarray(cal["Livox"]["Thermal_L"], np.float64)
    thermal2lidar = inverse_T(lidar2thermal)
    radar2lidar = thermal2lidar @ radar2thermal
    print("calibration['Radar']['Thermal_L']  (radar -> thermal camera)\n", radar2thermal)
    print("\ncalibration['Livox']['Thermal_L']  (Livox -> thermal camera)\n", lidar2thermal)
    print("\nthermal -> Livox = inverse of the one above\n", thermal2lidar)
    print("\nradar -> Livox = (thermal -> Livox) @ (radar -> thermal)\n", radar2lidar)
    print("""
How to read a 4x4 T = [[R, t], [0 0 0 1]]:
  * column 4 (t) = where the SOURCE sensor's origin lands in the TARGET frame, in metres
  * columns 1-3 (R) = where the source's x, y, z axes point, written in the target's axes""")
    t = radar2lidar[:3, 3]
    print("  -> the radar's origin is at Livox (x, y, z) = (%.2f, %.2f, %.2f) m: %.2f m %s, %.2f m %s, %.2f m %s the Livox"
          % (*t, abs(t[0]), "ahead of" if t[0] > 0 else "behind", abs(t[1]), "left of" if t[1] > 0 else "right of",
             abs(t[2]), "above" if t[2] > 0 else "below"))
    words = [("forward", "back"), ("left", "right"), ("up", "down")]
    for k, axis in enumerate("xyz"):
        v = radar2lidar[:3, k]
        m = int(np.argmax(np.abs(v)))
        print("  -> the radar's own %s axis points %-7s (%.1f deg off exactly %s)"
              % (axis, words[m][0] if v[m] > 0 else words[m][1], np.degrees(np.arccos(min(1, abs(v[m])))),
                 words[m][0] if v[m] > 0 else words[m][1]))
    print("  -> so the radar does not use 'x forward' like the Livox: its axes are relabelled (a 90 deg turn in the\n"
          "     matrix), plus a small physical tilt of a few degrees. The thermal camera uses a third convention:\n"
          "     x right, y down, z forward (distance in front of the lens), standard for cameras.")

    # ------------------------------------------------------------------------------------------------------------
    step(3, "Load this frame's radar points: they are in the RADAR's own frame (origin = the radar itself)")
    rel = info["sensor"]["radar_path"]
    raw = np.load(os.path.join(args.root, rel))
    print("file:", rel, "  fields:", raw.dtype.names, "  points:", len(raw))
    pts_radar = np.stack([raw["x"], raw["y"], raw["z"]], -1).astype(np.float64)
    print("first 5 points (x, y, z) in radar frame:\n", pts_radar[:5])

    # ------------------------------------------------------------------------------------------------------------
    boxes = np.asarray(info["annos"]["gt_boxes_livox"], np.float64)[:, :7]
    names = np.asarray(info["annos"]["name"])
    pts_code = (radar2lidar @ np.c_[pts_radar, np.ones(len(pts_radar))].T).T[:, :3]   # dsert_dataset.py:169
    hits = [(j, in_box(pts_code, b)) for j, b in enumerate(boxes)]
    j_best, mask = max(hits, key=lambda h: h[1].sum()) if hits else (None, np.zeros(len(pts_code), bool))
    i = int(np.flatnonzero(mask)[0]) if mask.any() else 0
    step(4, "Convert ONE point by hand (point %d, which lies on a labelled %s)" % (i, names[j_best] if mask.any() else "-"))
    p = np.r_[pts_radar[i], 1.0]
    print("radar frame      [x, y, z, 1] =", p, "   (the 1 lets one matrix both rotate AND shift)")
    p_th = radar2thermal @ p
    print("thermal frame    radar2thermal @ p =", p_th, "  (camera axes: x right, y down, z = distance in front)")
    p_li = thermal2lidar @ p_th
    print("Livox frame      thermal2lidar @ that =", p_li)
    print("one matrix       radar2lidar @ p =", radar2lidar @ p, "  (same thing in one go)")

    # ------------------------------------------------------------------------------------------------------------
    step(5, "Do all points the way the training code does, and check both code paths agree")
    single = (pts_radar @ radar2lidar[:3, :3].T + radar2lidar[:3, 3])      # dsert_single_sensor_dataset.py:163
    print("fusion-dataset form (dsert_dataset.py:169) vs single-sensor form (:163): max difference %.2e m"
          % np.abs(pts_code - single).max())
    print("first 5 points in Livox frame:\n", pts_code[:5])

    # ------------------------------------------------------------------------------------------------------------
    step(6, "Angles: each sensor measures the SAME object from ITS OWN position and forward direction")
    print("azimuth = angle from the sensor's forward axis, measured at the sensor (+ = left). Range = distance.")
    # Which radar axis is forward / left (from the matrix: the radar axis that maps onto Livox x / y).
    R = radar2lidar[:3, :3]
    f, l = int(np.argmax(np.abs(R[0]))), int(np.argmax(np.abs(R[1])))
    sf, sl = np.sign(R[0, f]), np.sign(R[1, l])
    print("radar forward = its %s%s axis, radar left = its %s%s axis"
          % ("-" if sf < 0 else "+", "xyz"[f], "-" if sl < 0 else "+", "xyz"[l]))
    print("%-12s %-30s %-34s %s" % ("object", "seen from Livox (range, az)", "seen from radar, raw (x, y)",
                                    "radar (range, az)"))
    lidar2radar = inverse_T(radar2lidar)
    for j, b in enumerate(boxes[:8]):
        c_r = lidar2radar @ np.r_[b[:3], 1]
        az_r = np.degrees(np.arctan2(sl * c_r[l], sf * c_r[f]))
        print("%-12s %6.2f m  %+7.2f deg             (%7.2f, %7.2f)                 %6.2f m  %+7.2f deg   (%d radar pts on it)"
              % (names[j], np.hypot(*b[:2]), azimuth(b), c_r[0], c_r[1], np.hypot(c_r[f], c_r[l]), az_r,
                 in_box(pts_code, b).sum()))
    print("Same object, different raw numbers; the calibration is what turns one into the other. Because radar and\n"
          "Livox sit within ~10 cm of each other here, range and angle are nearly equal once the axes are relabelled.")

    # ------------------------------------------------------------------------------------------------------------
    step(7, "Picture: radar before (own frame) and after (Livox frame) conversion, on top of the LiDAR and boxes")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib missing, no picture")
        return
    lraw = np.load(os.path.join(args.root, info["sensor"]["livox_path"]))
    lid = np.stack([lraw["x"], lraw["y"], lraw["z"]], -1)
    lid = lid[(lid[:, 2] > -2) & (lid[:, 2] < 4)]
    lid = lid[np.random.default_rng(0).choice(len(lid), min(len(lid), 50000), replace=False)]
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    for ax, pts, title in [(axes[0], pts_radar, "radar points in the RADAR's frame (not converted)"),
                           (axes[1], pts_code, "radar points after radar -> thermal -> Livox")]:
        ax.scatter(lid[:, 0], lid[:, 1], s=0.2, c="0.7", label="LiDAR (Livox frame)")
        ax.scatter(pts[:, 0], pts[:, 1], s=16, c="tab:blue", edgecolors="k", linewidths=.3, label="radar")
        for b in boxes:
            c = np.array([[1, 1], [1, -1], [-1, -1], [-1, 1], [1, 1]]) * b[3:5] / 2
            h = b[6]
            c = c @ np.array([[np.cos(h), np.sin(h)], [-np.sin(h), np.cos(h)]]) + b[:2]
            ax.plot(c[:, 0], c[:, 1], c="tab:red", lw=1.2)
        ax.plot(0, 0, "ks", ms=7, label="Livox origin")
        ax.plot(*radar2lidar[:2, 3], "k^", ms=8, label="radar position")
        far = max(30, np.hypot(boxes[:, 0], boxes[:, 1]).max() + 5) if len(boxes) else 50
        ax.set_xlim(-3, far)
        ax.set_ylim(-far / 2, far / 2)
        ax.set_aspect("equal")
        ax.set_xlabel("x forward (m)")
        ax.set_ylabel("y left (m)")
        ax.set_title(title)
        ax.legend(loc="lower right", markerscale=2, fontsize=8)
    fig.suptitle("%s  %s  frame %s  (red = labelled boxes)" % (meta.get("weather"), info.get("sequence_name", ""),
                                                               info.get("sample_idx", args.frame)))
    path = os.path.join(args.out, "%s_frame%d_before_after.png" % (args.weather, args.frame))
    fig.savefig(path, dpi=100, bbox_inches="tight")
    print("wrote", path)


if __name__ == "__main__":
    main()
