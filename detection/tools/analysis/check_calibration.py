"""
check_calibration.py: is the radar -> LiDAR calibration of DSERT-RoLL right?  CPU only, needs numpy + matplotlib
(Pillow for the camera pictures). It does not import the training code, so it runs on Kaggle, Snellius or a laptop.

WHAT "CALIBRATION" MEANS HERE
  Every sensor measures in its own coordinate system ("frame"): origin at the sensor, axes along how it is mounted.
  calibration[A][B] in label.pkl is a 4x4 matrix T that moves a point from frame A into frame B:

        [x_B, y_B, z_B, 1] = T @ [x_A, y_A, z_A, 1]        T = | R  t |    R: 3x3 rotation (how B is turned vs A)
                                                               | 0  1 |    t: shift in metres

  The labelled boxes live in the Livox LiDAR frame (x forward, y left, z up). label.pkl has no direct radar <-> Livox
  matrix, so training and evaluation chain two matrices through the left thermal camera:

        radar -> Livox = inverse(calibration['Livox']['Thermal_L']) @ calibration['Radar']['Thermal_L']
                         (thermal -> Livox)                          (radar -> thermal)

WHAT THIS SCRIPT CHECKS (each prints PASS / WARN and what it means)
  1. The sensor graph: which matrices label.pkl has.
  2. Every 4x4 is a real rigid motion: R is a pure rotation (R^T R = I, det +1), last row 0 0 0 1, shift of metres.
  3. Loops close: going A -> B -> C must land where A -> C lands, for every triangle of stored matrices.
     That proves the matrices agree with each other, not that they agree with the real car.
  4. Same rig in every drive: the sensors are bolted to one car, so radar -> Livox should not change between drives.
  5. The data test, the one that matters: radar points should land on the labelled objects (which were drawn on the
     LiDAR). We add a deliberate error to the calibration (turn the radar by -6..+6 deg, shift it by -1.5..+1.5 m)
     and count the radar points that land in labelled boxes. If the calibration is right, the count is highest at
     zero added error. If it peaks at +2 deg, the stored calibration is about 2 deg off.
  6. Pictures: bird's-eye view of a frame as calibrated next to the same frame with a deliberate 3 deg / 0.5 m
     error, and the camera image with LiDAR, radar and boxes drawn on it (that one also checks Livox -> camera).

Usage
  python detection/tools/analysis/check_calibration.py --data_root /kaggle/input --out_dir /kaggle/working/calib_check
  python detection/tools/analysis/check_calibration.py --data_root detection/data/dsert-roll --num_frames 300
"""
import argparse
import itertools
import os
import re
import sys

try:
    import pickle5 as pickle  
    # Python < 3.8 needs the protocol-5 backport, like the training code
except ImportError:
    import pickle

import numpy as np

DRIVE = re.compile(r"\d{4}(_\d{2}){5}$")          # drive folders are named like 2025_04_16_07_04_30
CLASSES = ("Vehicle", "Pedestrian", "Bike")       # the classes the models are trained and scored on
FEATURE_FALLBACKS = ["intensity", "reflectivity", "signal", "doppler"]
RESULTS = []                                      # (check, PASS/WARN, message) for the summary at the end


def report(check, ok, msg):
    RESULTS.append((check, "PASS" if ok else "WARN", msg))
    print("  [%s] %s" % ("PASS" if ok else "WARN", msg))


def section(title, explain):
    print("\n" + "=" * 100 + "\n" + title + "\n" + "=" * 100)
    for line in explain.strip().splitlines():
        print("  " + line.strip())
    print()


# ----------------------------------------------------------------------------------------------- finding the data
def find_data(data_root):
    """Drive folders and their label.pkl, and the processed_data folders that hold them (data may be split over
    several Kaggle datasets, so a file listed in label.pkl is looked up in each of them)."""
    roots, labels = [], {}
    for path, dirs, files in os.walk(data_root, followlinks=True):
        dirs[:] = [d for d in dirs if not d.startswith(".")]   # skip e.g. .cache left by the Hugging Face download
        for d in [d for d in dirs if DRIVE.match(d) or os.path.exists(os.path.join(path, d, "label.pkl"))]:
            dirs.remove(d)                        # do not walk into a drive's thousands of frame files
            root = os.path.dirname(path)          # path is .../processed_data/<Weather>
            if root not in roots:
                roots.append(root)
            if os.path.exists(os.path.join(path, d, "label.pkl")):
                labels.setdefault(d, os.path.join(path, d, "label.pkl"))
    return roots, labels


def resolve(roots, rel):
    for r in roots:
        p = os.path.join(r, rel)
        if os.path.exists(p):
            return p
    return ""


def load_label(path):
    with open(path, "rb") as f:
        z = pickle.load(f)
    infos = []
    for inf in z["info"]:
        inf = dict(inf)
        inf.update(z.get("meta", {}))             # calibration, weather and light are stored once per drive
        infos.append(inf)
    return infos


def load_points(path, feature):
    """(N, 4) [x, y, z, feature] in the sensor's own frame, as DSERTSingleSensorDataset.load_sensor_points reads it."""
    raw = np.load(path)
    if raw.size == 0:
        return np.zeros((0, 4))
    if raw.dtype.names is not None:
        name = next((f for f in [feature] + FEATURE_FALLBACKS if f in raw.dtype.names), "")
        pts = np.stack([raw["x"], raw["y"], raw["z"], raw[name] if name else np.zeros(raw.shape[0])], -1)
    else:
        arr = raw.reshape(-1, raw.shape[-1])
        pts = np.c_[arr[:, :3], arr[:, 3] if arr.shape[1] > 3 else np.zeros(len(arr))]
    pts = pts.astype(np.float64)
    return pts[np.isfinite(pts).all(1) & (np.abs(pts[:, :3]).sum(1) > 0)]


def labelled_boxes(info, max_dist):
    a = info.get("annos")
    if not a or len(a["name"]) == 0:
        return np.zeros((0, 7))
    boxes, names = np.asarray(a["gt_boxes_livox"], np.float64)[:, :7], np.asarray(a["name"])
    keep = np.isin(names, CLASSES) & (np.hypot(boxes[:, 0], boxes[:, 1]) < max_dist)
    return boxes[keep]


# ----------------------------------------------------------------------------------------------- 4x4 transforms
def as_T(m):
    try:
        T = np.asarray(m, np.float64)
    except (TypeError, ValueError):
        return None
    return T if T.shape == (4, 4) else None


def inverse_T(T):
    T_inv = np.eye(4)
    T_inv[:3, :3] = np.linalg.inv(T[:3, :3])
    T_inv[:3, 3] = -T_inv[:3, :3] @ T[:3, 3]
    return T_inv


def stored_extrinsics(cal):
    """{(A, B): 4x4} for every A -> B matrix in the calibration dict."""
    out = {}
    for a, row in cal.items():
        if isinstance(row, dict):
            for b, m in row.items():
                T = as_T(m)
                if T is not None:
                    out[a, b] = T
    return out


def radar_to_lidar(cal):
    """radar -> Livox, built exactly like DSERTSingleSensorDataset.sensor_to_livox (direct if stored, else through a
    camera both are calibrated to, preferring Thermal_L like the fusion dataset)."""
    E = stored_extrinsics(cal)
    if ("Radar", "Livox") in E:
        return E["Radar", "Livox"], "calibration['Radar']['Livox']"
    if ("Livox", "Radar") in E:
        return inverse_T(E["Livox", "Radar"]), "inverse(calibration['Livox']['Radar'])"
    common = [b for (a, b) in E if a == "Radar" and ("Livox", b) in E]
    if not common:
        raise KeyError("no route radar -> Livox in the calibration (keys: %s)" % sorted(cal))
    via = "Thermal_L" if "Thermal_L" in common else common[0]
    return (inverse_T(E["Livox", via]) @ E["Radar", via],
            "inverse(calibration['Livox']['%s']) @ calibration['Radar']['%s']" % (via, via))


def angle_deg(R):
    """Size of a rotation in degrees (0 = no rotation)."""
    return np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))


def diff(T1, T2):
    """How far apart two transforms are: (rotation in degrees, shift in cm)."""
    return angle_deg(T1[:3, :3] @ T2[:3, :3].T), 100 * np.linalg.norm(T1[:3, 3] - T2[:3, 3])


def yaw_pitch_roll(R):
    """R = Rz(yaw) Ry(pitch) Rx(roll), in degrees."""
    return (np.degrees(np.arctan2(R[1, 0], R[0, 0])), np.degrees(-np.arcsin(np.clip(R[2, 0], -1, 1))),
            np.degrees(np.arctan2(R[2, 1], R[2, 2])))


def rot_z(deg):
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def rot_y(deg):
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


# ----------------------------------------------------------------------------------------------- boxes
def in_any_box(p, boxes, margin, use_z):
    """(N,) True where point p[i] is inside one of the boxes [x, y, z, dx, dy, dz, heading], grown by `margin` m."""
    if len(boxes) == 0 or len(p) == 0:
        return np.zeros(len(p), bool)
    d = p[None, :, :3] - boxes[:, None, :3]                         # (M, N, 3)
    c, s = np.cos(-boxes[:, 6:7]), np.sin(-boxes[:, 6:7])
    lx, ly = d[..., 0] * c - d[..., 1] * s, d[..., 0] * s + d[..., 1] * c
    inside = (np.abs(lx) <= boxes[:, 3:4] / 2 + margin) & (np.abs(ly) <= boxes[:, 4:5] / 2 + margin)
    if use_z:
        inside &= np.abs(d[..., 2]) <= boxes[:, 5:6] / 2 + margin
    return inside.any(0)


def box_corners(b):
    x, y, z, dx, dy, dz, h = b[:7]
    c = np.array([[1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
                  [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1]]) * [dx / 2, dy / 2, dz / 2]
    return c @ rot_z(np.degrees(h)).T + [x, y, z]


BOX_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]


# ----------------------------------------------------------------------------------------------- checks 1-4
def check_graph(cal):
    section("1. Which calibration matrices does label.pkl have?",
            """calibration[A][B] = 4x4 matrix A -> B. 'intrinsic' / 'shape' entries are camera lenses (focal length,
            image centre, image size); they only matter for drawing on images.""")
    for a in sorted(cal, key=str):
        row = cal[a]
        if not isinstance(row, dict):
            print("  %-12s %s" % (a, type(row).__name__))
            continue
        ext = [b for b in row if as_T(row[b]) is not None]
        other = [b for b in row if b not in ext]
        print("  %-12s -> %s%s" % (a, ", ".join(map(str, ext)) or "(no 4x4)",
                                   ("   | also: " + ", ".join(map(str, other))) if other else ""))


def check_rigid(cal):
    section("2. Is every matrix a real rigid motion (a rotation plus a shift)?",
            """A rotation matrix must satisfy R^T R = I and det(R) = +1. det = -1 would mirror the world (a swapped
            axis); scale or shear would stretch it. The last row must be 0 0 0 1. Sensors on one car are at most a
            few metres apart, so the shift |t| should be small.""")
    bad = []
    for (a, b), T in sorted(stored_extrinsics(cal).items(), key=str):
        R, t = T[:3, :3], T[:3, 3]
        ortho = np.abs(R.T @ R - np.eye(3)).max()
        det = np.linalg.det(R)
        last = np.abs(T[3] - [0, 0, 0, 1]).max()
        ok = ortho < 1e-3 and abs(det - 1) < 1e-3 and last < 1e-6 and np.linalg.norm(t) < 5
        print("  %-10s -> %-10s |R^T R - I| = %.1e  det = %+.4f  |t| = %.2f m  %s"
              % (a, b, ortho, det, np.linalg.norm(t), "" if ok else "<-- odd"))
        if not ok:
            bad.append("%s->%s" % (a, b))
    report("rigid", not bad, "all matrices are rigid motions" if not bad else "odd matrices: %s" % ", ".join(bad))


def check_loops(cal):
    section("3. Do the matrices agree with each other (loop closure)?",
            """For three sensors A, B, C: going A -> B -> C must give the same as the stored A -> C. Each matrix is
            used as stored or inverted. Agreement only shows the numbers are consistent with each other (they may
            even have been derived from each other); check 5 is what compares them with the real car.""")
    E = stored_extrinsics(cal)
    get = lambda a, b: E.get((a, b), inverse_T(E[b, a]) if (b, a) in E else None)
    pairs = {frozenset(k) for k in E if k[0] != k[1]}
    nodes = sorted({n for k in E for n in k}, key=str)
    loops = []
    for a, b, c in itertools.combinations(nodes, 3):
        if {frozenset((a, b)), frozenset((b, c)), frozenset((a, c))} <= pairs:
            rot, cm = diff(get(b, c) @ get(a, b), get(a, c))
            loops.append((rot, cm, "%s -> %s -> %s" % (a, b, c)))
    # Radar and Livox are each calibrated to every camera, so radar -> Livox can be built through any of them
    # (radar -> camera -> Livox). All routes must give the same answer.
    routes = [b for (a, b) in E if a == "Radar" and ("Livox", b) in E]
    if len(routes) > 1:
        ref = inverse_T(E["Livox", "Thermal_L" if "Thermal_L" in routes else routes[0]])
        ref = ref @ E["Radar", "Thermal_L" if "Thermal_L" in routes else routes[0]]
        for via in routes:
            rot, cm = diff(inverse_T(E["Livox", via]) @ E["Radar", via], ref)
            loops.append((rot, cm, "Radar -> %s -> Livox (vs via Thermal_L)" % via))
    if not loops:
        print("  no triangle of matrices to compare (each pair of sensors is connected by one route only)")
        return
    loops.sort(key=lambda l: -(l[0] + l[1] / 10))
    radar = [l for l in loops if "Radar" in l[2]]
    show = loops[:5] + [l for l in radar if l not in loops[:5]]
    for rot, cm, name in show:
        print("  %-40s differs from the direct matrix by %.3f deg and %.1f cm" % (name, rot, cm))
    worst = loops[0]
    report("loops", worst[0] < 0.5 and worst[1] < 5,
           "%d loops checked, worst %.3f deg / %.1f cm (%s)" % (len(loops), worst[0], worst[1], worst[2]))


def describe_radar(T, how):
    section("Radar -> Livox (the transform every radar point goes through)",
            """Livox frame: x forward, y left, z up. t is where the radar sits seen from the Livox; the angles are how
            the radar's axes are turned relative to the Livox's.""")
    print("  built as: " + how)
    print("  " + np.array2string(T, precision=4, suppress_small=True).replace("\n", "\n  "))
    t = T[:3, 3]
    print("\n  the radar sits %.2f m %s, %.2f m %s and %.2f m %s the Livox"
          % (abs(t[0]), "in front of" if t[0] >= 0 else "behind", abs(t[1]), "left of" if t[1] >= 0 else "right of",
             abs(t[2]), "above" if t[2] >= 0 else "below"))
    yaw, pitch, roll = yaw_pitch_roll(T[:3, :3])
    print("  rotation: yaw %+.2f deg (+ = turned left), pitch %+.2f deg (+ = nose down), roll %+.2f deg" % (yaw, pitch, roll))
    words = [("forward", "back"), ("left", "right"), ("up", "down")]
    for i, axis in enumerate("xyz"):
        v = T[:3, i]
        k = int(np.argmax(np.abs(v)))
        off = np.degrees(np.arccos(np.clip(abs(v[k]), -1, 1)))
        print("  the radar's own %s axis points %s (%.1f deg off the Livox's %s axis)"
              % (axis, words[k][0] if v[k] > 0 else words[k][1], off, "xyz"[k]))


def check_drives(drive_cal):
    section("4. Is radar -> Livox the same in every drive?",
            """All drives were recorded with the same sensors bolted to the same car, so the matrix should be
            identical, or differ by a hair if the rig was re-calibrated between recording days.""")
    names = sorted(drive_cal)
    ref = drive_cal[names[0]][0]
    rows = [(n, *diff(drive_cal[n][0], ref), drive_cal[n][1]) for n in names]
    uniq = []
    for n, rot, cm, w in rows:
        if not any(np.allclose(drive_cal[n][0], u, atol=1e-9) for u in uniq):
            uniq.append(drive_cal[n][0])
    print("  %d drives, %d different radar -> Livox matrices" % (len(names), len(uniq)))
    for n, rot, cm, w in sorted(rows, key=lambda r: -(r[1] + r[2] / 10))[:8]:
        print("  %-22s %-11s %.3f deg / %.1f cm from drive %s" % (n, w, rot, cm, names[0]))
    worst = max(rows, key=lambda r: r[1] + r[2] / 10)
    report("drives", worst[1] < 0.5 and worst[2] < 5,
           "largest difference between drives: %.3f deg / %.1f cm (%s)" % (worst[1], worst[2], worst[0]))


# ----------------------------------------------------------------------------------------------- check 5
def collect_frames(all_infos, roots, n_frames, max_dist, rng):
    """Up to n_frames frames, spread over the drives, with labelled boxes and radar points (in the Livox frame)."""
    cand = [(d, inf) for d, infs in all_infos.items() for inf in infs if len(labelled_boxes(inf, max_dist))]
    if not cand:
        return []
    order = rng.permutation(len(cand))
    frames = []
    for i in order:
        if len(frames) >= n_frames:
            break
        drive, info = cand[i]
        p = resolve(roots, info["sensor"].get("radar_path", ""))
        if not p:
            continue
        rad = load_points(p, "doppler")
        T, _ = radar_to_lidar(info["calibration"])
        rad[:, :3] = rad[:, :3] @ T[:3, :3].T + T[:3, 3]
        rad = rad[(rad[:, 0] > 0) & (np.hypot(rad[:, 0], rad[:, 1]) < max_dist)]
        if len(rad) >= 3:
            frames.append(dict(drive=drive, info=info, pts=rad, boxes=labelled_boxes(info, max_dist), origin=T[:3, 3]))
    return frames


def apply_error(fr, e):
    """The frame's radar points with an extra calibration error e = {yaw, pitch (deg), x, y, z (m)}: the radar is
    turned around its own position, then shifted. e['dt'] (s) is a timing error, not a calibration one: each point
    moves along its line of sight by doppler * dt, i.e. where it would be dt seconds later."""
    p, o = fr["pts"][:, :3], fr["origin"]
    out = (p - o) @ (rot_z(e["yaw"]) @ rot_y(e["pitch"])).T + o + [e["x"], e["y"], e["z"]]
    if e.get("dt", 0.0):
        los = p - o
        los /= np.maximum(np.linalg.norm(los, axis=1, keepdims=True), 1e-6)
        out = out + los * (fr["pts"][:, 3:4] * e["dt"])
    return out


def sweep(frames, kind, values, margin, others):
    """Fraction of radar points inside a labelled box when error `kind` takes each of `values` and the other
    errors stay at `others`."""
    out = []
    for v in values:
        e = dict(others, **{kind: v})
        hit = tot = 0
        for fr in frames:
            p = apply_error(fr, e)
            hit += in_any_box(p, fr["boxes"], margin, use_z=kind in ("z", "pitch")).sum()
            tot += len(p)
        out.append(hit / max(tot, 1))
    return np.array(out)


def peak(values, scores):
    """Centre of the top of the curve (offsets scoring >= 97% of the best, contiguous with the best)."""
    i = int(np.argmax(scores))
    lo = hi = i
    while lo > 0 and scores[lo - 1] >= 0.97 * scores[i]:
        lo -= 1
    while hi < len(scores) - 1 and scores[hi + 1] >= 0.97 * scores[i]:
        hi += 1
    return float(values[lo:hi + 1].mean()), lo == 0 or hi == len(scores) - 1


SWEEPS = [  # name, offsets, unit, tolerance, what a positive / negative best offset means
    ("dt", np.arange(-0.25, 0.2501, 0.01), "s", 0.03, ("captured earlier than the LiDAR", "captured later")),
    ("yaw", np.arange(-6, 6.001, 0.25), "deg", 1.0, ("turned left", "turned right")),
    ("x", np.arange(-2, 2.001, 0.1), "m", 0.3, ("moved forward", "moved back")),
    ("y", np.arange(-1.5, 1.501, 0.1), "m", 0.3, ("moved left", "moved right")),
    ("z", np.arange(-1.5, 1.501, 0.1), "m", 0.4, ("moved up", "moved down")),
    ("pitch", np.arange(-6, 6.001, 0.5), "deg", 1.5, ("tilted nose-down", "tilted nose-up")),
]


def check_data(frames, margin, out_dir, rng):
    section("5. The data test: do the radar points land on the labelled objects?",
            """The boxes were drawn on the LiDAR, so they tell where the objects really are in the Livox frame. We
            add a deliberate error to the radar calibration and count the radar points inside the boxes (grown by
            %.2f m because radar is noisy). Right calibration = the count peaks at zero added error. yaw / x / y are
            judged from above (bird's-eye view); z / pitch also need the point inside the box's height, which is
            weak for radar because its elevation is coarse, so those curves are flatter.
            dt is NOT a calibration error but a timing one: if the radar scan was taken dt seconds before the LiDAR
            scan the boxes were drawn on, a moving object has moved by (its speed toward/away from us) * dt in the
            meantime. The radar measures that speed per point (Doppler), so dt is fitted with the calibration errors;
            otherwise it would show up as a fake forward / backward calibration error.""" % margin)
    n_pts = sum(len(f["pts"]) for f in frames)
    n_box = sum(len(f["boxes"]) for f in frames)
    print("  %d frames from %d drives, %d radar points within range, %d labelled boxes"
          % (len(frames), len({f["drive"] for f in frames}), n_pts, n_box))

    # Chance level: the same radar points scored against the boxes of a different frame.
    perm = rng.permutation(len(frames))
    hit = sum(in_any_box(f["pts"][:, :3], frames[j]["boxes"], margin, False).sum() for f, j in zip(frames, perm))
    chance = hit / max(n_pts, 1)
    zero = {name: 0.0 for name, *_ in SWEEPS}
    base = sweep(frames, "x", [0.0], margin, zero)[0]
    print("  radar points inside a labelled box: %.1f%% as calibrated, %.1f%% against another frame's boxes (chance)"
          % (100 * base, 100 * chance))
    report("data: above chance", base > 2 * chance,
           "radar points hit the boxes %.1fx more often than chance" % (base / max(chance, 1e-9)))

    # Which share of boxes gets any radar point, by distance (cars should; distant pedestrians often don't).
    bins = [0, 15, 30, 50, 80]
    got, tot = np.zeros(len(bins) - 1), np.zeros(len(bins) - 1)
    for f in frames:
        dist = np.hypot(f["boxes"][:, 0], f["boxes"][:, 1])
        for j, b in enumerate(f["boxes"]):
            k = np.searchsorted(bins, dist[j], side="right") - 1
            if 0 <= k < len(got):
                tot[k] += 1
                got[k] += in_any_box(f["pts"][:, :3], b[None], margin, False).any()
    print("  labelled boxes with >= 1 radar point: " + "   ".join(
        "%d-%dm %d/%d" % (bins[k], bins[k + 1], got[k], tot[k]) for k in range(len(got)) if tot[k]))

    # Errors mix (a wrong turn also moves far points sideways), so find the best correction for all five together:
    # sweep one while the others stay at their current best, three rounds. The last round's curves are plotted.
    best = dict(zero)
    for _ in range(3):
        curves = {}
        for name, values, unit, tol, meaning in SWEEPS:
            scores = sweep(frames, name, values, margin, best)
            best[name], at_edge = peak(values, scores)
            curves[name] = (values, scores, best[name], unit, at_edge)
    fixed = sweep(frames, "x", [best["x"]], margin, best)[0]
    print("\n  best correction, all fitted together (radar points in a box: %.1f%% -> %.1f%% after correcting)"
          % (100 * base, 100 * fixed))
    print("  %-6s %-14s %-12s %s" % ("error", "best at", "tolerance", "verdict"))
    for name, values, unit, tol, meaning in SWEEPS:
        b, at_edge = curves[name][2], curves[name][4]
        ok = abs(b) <= tol and not at_edge
        if name == "dt":
            verdict = ("radar and LiDAR in sync" if ok else
                       "TIMING, not calibration: the radar scan looks %.0f ms %s; a car closing at 20 m/s is "
                       "then drawn %.1f m off its box" % (1000 * abs(b), meaning[0] if b > 0 else meaning[1],
                                                         20 * abs(b)))
        else:
            verdict = ("matches the labels" if ok else "fits the labels better with the radar %s by %.2f %s%s"
                       % (meaning[0] if b > 0 else meaning[1], abs(b), unit,
                          " (or more: the best is at the end of the tested range)" if at_edge else ""))
        print("  %-6s %+8.2f %-5s +-%-4.2g %-5s %s" % (name, b, unit, tol, unit, verdict))
        report("data: " + name, ok, "%s best at %+.2f %s (tolerance %.2g %s)%s"
               % (name, b, unit, tol, unit, " = radar/LiDAR timing gap" if name == "dt" else ""))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib missing: no plots")
        return
    fig, axes = plt.subplots(1, len(curves), figsize=(4 * len(curves), 3.4))
    for ax, (name, (values, scores, b, unit, _)) in zip(axes, curves.items()):
        ax.plot(values, 100 * scores, "-o", ms=3)
        ax.axvline(0, c="k", lw=1, label="as calibrated")
        ax.axvline(b, c="tab:red", ls="--", lw=1, label="best %+.2f" % b)
        if name in ("yaw", "x", "y"):
            ax.axhline(100 * chance, c="0.6", ls=":", lw=1, label="chance")
        ax.set_xlabel("added %s error (%s)" % (name, unit))
        ax.set_title(name)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("% radar points inside a box")
    fig.suptitle("Peak at 0 = calibration agrees with the labels (%d frames)" % len(frames))
    fig.tight_layout()
    path = os.path.join(out_dir, "sweeps.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print("\n  wrote " + path)


# ----------------------------------------------------------------------------------------------- check 6: pictures
def draw_frames(frames, roots, out_dir, n, rng, pc_range):
    section("6. Pictures",
            """bev_*.png: from above. Grey = LiDAR, coloured = radar (colour = Doppler speed), red = labelled boxes.
            Left as calibrated, right with a deliberate 3 deg turn + 0.5 m shift, to show what a wrong calibration
            looks like: the radar clusters slide off the boxes, more so the farther away they are.
            camera_*.png: the left RGB image with LiDAR (small, colour = distance), radar (big circles) and the
            boxes drawn on it. LiDAR edges should sit on the object edges (checks Livox -> camera); radar circles
            should sit on the cars (checks the whole chain radar -> thermal -> Livox -> camera).""")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib missing: no pictures")
        return
    # Frames with the most radar points inside boxes are the clearest to look at.
    score = [in_any_box(f["pts"][:, :3], f["boxes"], 0.25, False).sum() for f in frames]
    picks = [frames[i] for i in np.argsort(score)[::-1][:n]]
    for k, fr in enumerate(picks):
        info = fr["info"]
        tag = "%s_%s" % (fr["drive"], info.get("sample_idx", k))
        lp = resolve(roots, info["sensor"].get("livox_path", ""))
        lid = load_points(lp, "intensity") if lp else np.zeros((0, 4))
        if len(lid) > 60000:
            lid = lid[rng.choice(len(lid), 60000, replace=False)]
        lid_bev = lid[(lid[:, 2] > pc_range[2]) & (lid[:, 2] < pc_range[5])]
        far = max(20.0, np.hypot(fr["boxes"][:, 0], fr["boxes"][:, 1]).max() + 5)

        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        wrong = (fr["pts"][:, :3] - fr["origin"]) @ rot_z(3).T + fr["origin"] + [0.5, 0, 0]
        for ax, pts, title in [(axes[0], fr["pts"][:, :3], "as calibrated"),
                               (axes[1], wrong, "with a deliberate +3 deg yaw, +0.5 m x error")]:
            ax.scatter(lid_bev[:, 0], lid_bev[:, 1], s=0.2, c="0.7")
            sc = ax.scatter(pts[:, 0], pts[:, 1], s=14, c=fr["pts"][:, 3], cmap="coolwarm", edgecolors="k",
                            linewidths=0.3)
            for b in fr["boxes"]:
                c = box_corners(b)[[0, 1, 2, 3, 0], :2]
                ax.plot(c[:, 0], c[:, 1], c="tab:red", lw=1.2)
            ax.plot(*fr["origin"][:2], "k^", ms=8)
            ax.plot(0, 0, "ks", ms=6)
            ax.set_xlim(-2, far)
            ax.set_ylim(-far / 2, far / 2)
            ax.set_aspect("equal")
            ax.set_title(title)
            ax.set_xlabel("x forward (m)")
            ax.set_ylabel("y left (m)")
        fig.colorbar(sc, ax=axes, shrink=0.6, label="radar Doppler")
        fig.suptitle("%s  %s / %s   (black square = Livox, black triangle = radar)"
                     % (tag, info.get("weather", ""), info.get("light", "")), y=0.93)
        path = os.path.join(out_dir, "bev_%d_%s.png" % (k, tag))
        fig.savefig(path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        print("  wrote " + path)

        draw_camera(fr, lid, roots, out_dir, k, tag, plt)


def draw_camera(fr, lid, roots, out_dir, k, tag, plt, cam="RGB_L"):
    info, cal = fr["info"], fr["info"]["calibration"]
    ip = resolve(roots, info["sensor"].get("rgb_left_path", ""))
    T = as_T(cal.get("Livox", {}).get(cam)) if isinstance(cal.get("Livox"), dict) else None
    intr = cal.get(cam, {}).get("intrinsic") if isinstance(cal.get(cam), dict) else None
    if not ip or T is None or intr is None:
        print("  no camera picture for %s (image %s, Livox->%s %s, intrinsic %s)"
              % (tag, "found" if ip else "missing", cam, "found" if T is not None else "missing",
                 "found" if intr is not None else "missing"))
        return
    try:
        from PIL import Image
    except ImportError:
        print("  Pillow missing: no camera pictures")
        return
    img = Image.open(ip).convert("RGB")
    K = np.asarray(intr, np.float64)[:3, :3].copy()
    shape = cal[cam].get("shape")
    cw, ch = (float(shape[0]), float(shape[1])) if shape is not None else img.size   # size the K was made for
    K[0] *= img.size[0] / cw
    K[1] *= img.size[1] / ch

    def project(xyz):
        p = np.c_[xyz, np.ones(len(xyz))] @ T.T
        z = p[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            uv = (p[:, :3] @ K.T)[:, :2] / z[:, None]
        ok = (z > 0.5) & (uv[:, 0] >= 0) & (uv[:, 0] < img.size[0]) & (uv[:, 1] >= 0) & (uv[:, 1] < img.size[1])
        return uv, z, ok

    fig, ax = plt.subplots(figsize=(14, 14 * img.size[1] / img.size[0]))
    ax.imshow(img)
    uv, z, ok = project(lid[:, :3])
    ax.scatter(uv[ok, 0], uv[ok, 1], s=0.4, c=z[ok], cmap="turbo", vmin=0, vmax=50)
    uv, z, ok = project(fr["pts"][:, :3])
    ax.scatter(uv[ok, 0], uv[ok, 1], s=60, facecolors="none", edgecolors="red", linewidths=1.5)
    for b in fr["boxes"]:
        uv, z, _ = project(box_corners(b))
        for i, j in BOX_EDGES:
            if z[i] > 0.5 and z[j] > 0.5:
                ax.plot(uv[[i, j], 0], uv[[i, j], 1], c="lime", lw=1)
    ax.set_xlim(0, img.size[0])
    ax.set_ylim(img.size[1], 0)
    ax.axis("off")
    ax.set_title("%s: LiDAR (dots, colour = distance), radar (red circles), labelled boxes (green)" % tag)
    path = os.path.join(out_dir, "camera_%d_%s.png" % (k, tag))
    fig.savefig(path, dpi=90, bbox_inches="tight")
    plt.close(fig)
    print("  wrote " + path)


# ----------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    laptop_sample = os.path.join(os.path.expanduser("~"), "dsert-roll-sample")   # one Fog + one Light_Snow drive
    default_root = next((p for p in ["/kaggle/input", laptop_sample] if os.path.isdir(p)), "detection/data/dsert-roll")
    ap.add_argument("--data_root", default=default_root, help="folder that contains processed_data/<Weather>/<drive>/")
    ap.add_argument("--out_dir", default="calib_check")
    ap.add_argument("--num_frames", type=int, default=200, help="frames used by the data test")
    ap.add_argument("--num_pictures", type=int, default=3)
    ap.add_argument("--max_dist", type=float, default=50.0, help="only boxes / radar points this close (m)")
    ap.add_argument("--margin", type=float, default=0.25, help="grow the boxes by this much for radar noise (m)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    pc_range = np.array([0, -76.8, -2, 76.8, 76.8, 4])           # POINT_CLOUD_RANGE of lidar_only.yaml

    roots, labels = find_data(args.data_root)
    if not labels:
        sys.exit("no label.pkl under %s (expected processed_data/<Weather>/<drive>/label.pkl)" % args.data_root)
    print("%d drives with label.pkl; data folders: %s" % (len(labels), roots))
    all_infos = {d: load_label(p) for d, p in sorted(labels.items())}
    first = next(inf for infs in all_infos.values() for inf in infs)
    cal = first["calibration"]

    check_graph(cal)
    check_rigid(cal)
    check_loops(cal)
    T, how = radar_to_lidar(cal)
    describe_radar(T, how)
    drive_cal = {}
    for d, infs in all_infos.items():
        if infs:
            drive_cal[d] = (radar_to_lidar(infs[0]["calibration"])[0], infs[0].get("weather", ""))
    check_drives(drive_cal)

    frames = collect_frames(all_infos, roots, args.num_frames, args.max_dist, rng)
    if not frames:
        report("data", False, "no frame with both labelled boxes and radar files found; data test skipped")
    else:
        check_data(frames, args.margin, args.out_dir, rng)
        draw_frames(frames, roots, args.out_dir, args.num_pictures, rng, pc_range)

    print("\n" + "=" * 100 + "\nSUMMARY\n" + "=" * 100)
    for check, status, msg in RESULTS:
        print("  [%s] %-18s %s" % (status, check, msg))
    n_warn = sum(s == "WARN" for _, s, _ in RESULTS)
    print("\n  " + ("everything passed: the radar calibration agrees with itself and with the labelled objects"
                    if not n_warn else "%d warning(s): read the sections above and look at the pictures" % n_warn))


if __name__ == "__main__":
    main()
