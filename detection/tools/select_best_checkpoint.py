"""
Pick the best checkpoint of a run from its per-epoch validation logs (written by test.py / eval_epochs.sbatch).

Score per epoch = mean Level-1 AP over Vehicle, Pedestrian and Cyclist (Bike) in the overall block of the newest
log_eval_*.txt under <run_dir>/eval/epoch_<N>/val/. Prints a table, writes <run_dir>/best_checkpoint.json, and prints
the chosen checkpoint path as the last line (for scripts). Fails if no epoch has those AP values.

    python select_best_checkpoint.py --run_dir ../output/det_model_cfgs/dsert/lidar_only/livox_pretrain --epochs 16 17 18 19 20
"""
import argparse
import glob
import json
import os
import re
import sys

AP_KEY = re.compile(r'TYPE_(VEHICLE|PEDESTRIAN|CYCLIST)_LEVEL_1/AP$')
LINE = re.compile(r'^\s*(\S+):\s*(-?\d+(?:\.\d+)?(?:[eE]-?\d+)?)\s*$')


def overall_block(log_path):
    """{metric: value} from the '==== Overall ====' block of an evaluation log."""
    values, inside = {}, False
    for line in open(log_path, errors='replace'):
        if '==== Overall ====' in line:
            inside = True
            continue
        if inside and '==== Per Weather ====' in line:
            break
        m = LINE.match(line) if inside else None
        if m:
            values[m.group(1)] = float(m.group(2))
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run_dir', required=True)
    parser.add_argument('--epochs', nargs='+', type=int, required=True)
    args = parser.parse_args()

    rows = []
    for e in args.epochs:
        logs = sorted(glob.glob(os.path.join(args.run_dir, 'eval', 'epoch_%d' % e, 'val', 'log_eval_*.txt')),
                      key=os.path.getmtime)
        if not logs:
            print('epoch %2d: no evaluation log' % e)
            continue
        aps = {AP_KEY.search(k).group(1): v for k, v in overall_block(logs[-1]).items() if AP_KEY.search(k)}
        if not aps:
            print('epoch %2d: no Level-1 AP values found in %s' % (e, logs[-1]))
            continue
        rows.append((e, sum(aps.values()) / len(aps), aps, logs[-1]))

    if not rows:
        sys.exit('ERROR: no epoch could be scored; check the evaluation logs under %s/eval' % args.run_dir)

    classes = ['VEHICLE', 'PEDESTRIAN', 'CYCLIST']
    print('\nepoch' + ''.join('%14s' % c for c in classes) + '%14s' % 'mean AP')
    for e, score, aps, _ in rows:
        print('%5d' % e + ''.join('%14s' % ('%.4f' % aps[c] if c in aps else '-') for c in classes) + '%14.4f' % score)
    best_epoch, best_score, best_aps, best_log = max(rows, key=lambda r: r[1])
    ckpt = os.path.abspath(os.path.join(args.run_dir, 'ckpt', 'checkpoint_epoch_%d.pth' % best_epoch))
    if not os.path.isfile(ckpt):
        sys.exit('ERROR: best epoch %d has no checkpoint file %s' % (best_epoch, ckpt))
    summary = {'best_epoch': best_epoch, 'mean_level1_ap': best_score, 'per_class_ap': best_aps, 'checkpoint': ckpt,
               'log': best_log, 'all_epochs': {str(e): {'mean_level1_ap': s, 'per_class_ap': a} for e, s, a, _ in rows}}
    with open(os.path.join(args.run_dir, 'best_checkpoint.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print('\nbest: epoch %d (mean Level-1 AP %.4f)' % (best_epoch, best_score))
    print(ckpt)


if __name__ == '__main__':
    main()
