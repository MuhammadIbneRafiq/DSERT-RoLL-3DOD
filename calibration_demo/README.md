# Calibration demo (runs on the laptop)

Sample data (downloaded from the public Hugging Face copy, `jeongyh98/DSERT-RoLL`, ~350 MB) lives outside OneDrive in
`C:\Users\wifi stuff\dsert-roll-sample\processed_data\`:

| weather | drive | frames | light | files |
|---|---|---|---|---|
| Fog | `2025_04_15_07_10_51` | 71 | Normal | `label.pkl`, radar, Livox, left thermal, left RGB |
| Light_Snow | `2025_02_12_04_40_15` | 145 | Low_Light | same |

## 1. Follow one frame through radar → thermal → Livox, number by number

```bash
python calibration_demo/radar_to_lidar_step_by_step.py --weather Fog --frame 10
python calibration_demo/radar_to_lidar_step_by_step.py --weather Light_Snow --frame 40
```

Picture: `calibration_demo/output/<weather>_frame<N>_before_after.png`.

## 2. Full calibration check (finds the laptop sample automatically)

```bash
python detection/tools/analysis/check_calibration.py --out_dir calibration_demo/output/calib_check
```

Pictures: `sweeps.png`, `bev_*.png`, `camera_*.png` in `calibration_demo/output/calib_check/`.

## Where the conversion happens in the training code

- `detection/al3d_det/datasets/dsert/dsert_dataset.py:164-169`: fusion model
- `detection/al3d_det/datasets/dsert/dsert_single_sensor_dataset.py:99-119` (builds the matrix) and `:162-163` (applies
  it): the `lidar_only.yaml` radar runs
