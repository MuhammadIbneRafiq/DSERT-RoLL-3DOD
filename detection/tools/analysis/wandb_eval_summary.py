"""
Send every evaluation that already exists in the log files to Weights & Biases. CPU only, runs on a login node.

For each experiment folder (detection/output/det_model_cfgs/dsert/lidar_only/<TAG>) it reads
  * eval/epoch_<N>/val/log_eval_*.txt  (test.py: per-epoch / standalone evaluations)
  * the newest log_train_*.txt         (the evaluation at the end of training)
and logs, into that experiment's own wandb run (resumed via its wandb_run_id.txt), the full AP breakdown against
'epoch': eval/overall/..., eval/per_weather/<W>/..., eval/per_light/<L>/..., plus eval/overall/mean_L1_AP.
It then creates one 'results_summary' run with comparison tables (final + best epoch per experiment, per-weather
vehicle AP) and a bar chart.

    python detection/tools/analysis/wandb_eval_summary.py --project dsert-single-sensor
"""
import argparse
import glob
import os
import re

RUNS = ['livox_pretrain', 'radar_scratch', 'radar_ft_from_livox', 'livox_e20_on_radar']
CLASSES = ['VEHICLE', 'PEDESTRIAN', 'CYCLIST']
BLOCK = re.compile(r'^\[(overall|weather=(.+)|light=(.+))\] N=(\d+)\s*$')
METRIC = re.compile(r'^(\S+):\s*(-?\d+(?:\.\d+)?(?:[eE]-?\d+)?)\s*$')
EPOCH = re.compile(r'Performance of EPOCH (\d+)')


def parse_results(path):
    """{flat metric name: value} from the last evaluation printout in a log file ({} if none)."""
    lines = open(path, errors='replace').read().splitlines()
    starts = [i for i, l in enumerate(lines) if '==== Overall ====' in l]
    if not starts:
        return {}, None
    epoch = None
    for l in lines[:starts[-1]]:
        m = EPOCH.search(l)
        if m:
            epoch = int(m.group(1))
    out, prefix = {}, None
    for l in lines[starts[-1] + 1:]:
        b = BLOCK.match(l)
        if b:
            prefix = 'overall' if b.group(1) == 'overall' else (
                'per_weather/' + b.group(2) if b.group(2) else 'per_light/' + b.group(3))
            continue
        m = METRIC.match(l)
        if m and prefix:
            out['%s/%s' % (prefix, m.group(1))] = float(m.group(2))
    l1 = [out.get('overall/OBJECT_TYPE_TYPE_%s_LEVEL_1/AP' % c) for c in CLASSES]
    if all(v is not None for v in l1):
        out['overall/mean_L1_AP'] = sum(l1) / len(l1)
    return out, epoch


def collect(run_dir):
    """{epoch: metrics} for one experiment; standalone evaluations override the end-of-training one."""
    results = {}
    train_logs = sorted(glob.glob(os.path.join(run_dir, 'log_train_*.txt')), key=os.path.getmtime)
    if train_logs:
        metrics, epoch = parse_results(train_logs[-1])
        if metrics and epoch is not None:
            results[epoch] = metrics
    for d in glob.glob(os.path.join(run_dir, 'eval', 'epoch_*', 'val')):
        m = re.search(r'epoch_(\d+)', d)
        logs = sorted(glob.glob(os.path.join(d, 'log_eval_*.txt')), key=os.path.getmtime)
        if m and logs:
            metrics, _ = parse_results(logs[-1])
            if metrics:
                results[int(m.group(1))] = metrics
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--project', default='dsert-single-sensor')
    parser.add_argument('--entity', default=None)
    parser.add_argument('--runs_dir', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..',
                                                          'output', 'det_model_cfgs', 'dsert', 'lidar_only'))
    parser.add_argument('--runs', nargs='+', default=RUNS)
    parser.add_argument('--dry_run', action='store_true', help='print what would be logged, do not contact wandb')
    args = parser.parse_args()

    all_results = {}
    for tag in args.runs:
        run_dir = os.path.abspath(os.path.join(args.runs_dir, tag))
        res = collect(run_dir)
        all_results[tag] = res
        print('%-22s epochs %s' % (tag, sorted(res) or 'none found'))
        for e in sorted(res):
            print('    epoch %2d  mean L1 AP %s  vehicle %s' % (e, '%.4f' % res[e]['overall/mean_L1_AP']
                  if 'overall/mean_L1_AP' in res[e] else '-',
                  res[e].get('overall/OBJECT_TYPE_TYPE_VEHICLE_LEVEL_1/AP', '-')))
    if args.dry_run:
        return

    import wandb
    for tag, res in all_results.items():
        id_file = os.path.join(args.runs_dir, tag, 'wandb_run_id.txt')
        if not res or not os.path.exists(id_file):
            print('skip %s: %s' % (tag, 'no results' if not res else 'no wandb_run_id.txt'))
            continue
        run = wandb.init(project=args.project, entity=args.entity, id=open(id_file).read().strip(), resume='allow')
        wandb.define_metric('epoch')
        wandb.define_metric('eval/*', step_metric='epoch')
        for e in sorted(res):
            wandb.log({**{'eval/' + k: v for k, v in res[e].items()}, 'epoch': e})
        run.finish()
        print('logged %d epochs to %s' % (len(res), tag))

    run = wandb.init(project=args.project, entity=args.entity, name='results_summary', job_type='summary')
    cols = ['experiment', 'epoch', 'which'] + ['%s_AP' % c.lower() for c in CLASSES] + ['mean_L1_AP']
    table = wandb.Table(columns=cols)
    weather_rows = []
    for tag, res in all_results.items():
        if not res:
            continue
        final = max(res)
        best = max(res, key=lambda e: res[e].get('overall/mean_L1_AP', -1))
        for e, which in sorted({(final, 'final'), (best, 'best')}):
            r = res[e]
            table.add_data(tag, e, which, *[r.get('overall/OBJECT_TYPE_TYPE_%s_LEVEL_1/AP' % c) for c in CLASSES],
                           r.get('overall/mean_L1_AP'))
        for k, v in res[final].items():
            m = re.match(r'per_weather/(.+)/OBJECT_TYPE_TYPE_VEHICLE_LEVEL_1/AP$', k)
            if m:
                weather_rows.append([tag, m.group(1), v])
    run.log({'results/ap_table': table,
             'results/vehicle_ap_by_weather': wandb.Table(columns=['experiment', 'weather', 'vehicle_AP'],
                                                          data=weather_rows),
             'results/mean_L1_AP_final': wandb.plot.bar(
                 wandb.Table(columns=['experiment', 'mean_L1_AP'],
                             data=[[t, all_results[t][max(all_results[t])].get('overall/mean_L1_AP')]
                                   for t in all_results if all_results[t]]),
                 'experiment', 'mean_L1_AP', title='mean Level-1 AP at the final epoch')})
    print('summary run:', getattr(run, 'url', None))
    run.finish()


if __name__ == '__main__':
    main()
