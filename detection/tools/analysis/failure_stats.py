"""
Statistical comparison of two models on the same labelled objects, from extract_features.py outputs. CPU only.

Both .npz files must come from the same frames (same --max_frames / --weather), so row i is the same object in both.
Detected = matched by a same-class prediction at the evaluation IoU (0.5 Vehicle, 0.3 Pedestrian/Bike).

  1. Overall: recall of each model, the difference, McNemar's exact test (paired: objects only one model detects),
     and a 95% bootstrap confidence interval that resamples whole sequences (frames within a sequence are correlated).
  2. Subgroups (weather, light, class, distance, points in box): the same paired test per subgroup, with Holm
     correction within each grouping, so testing many subgroups does not produce false positives by chance.
  3. Why objects are missed: logistic regression of detected on log(points in box), distance, weather and light per
     model (odds ratios, sequence-clustered standard errors), plus an interaction model testing where the second
     model's advantage grows or shrinks. Needs statsmodels (pip install statsmodels); skipped if missing.

These tests measure uncertainty from the validation data (which scenes and objects), not from training randomness:
repeated training seeds are still needed to claim one training recipe beats another.

    python analysis/failure_stats.py --baseline feats_radar_scratch.npz --treatment feats_radar_ft_from_livox.npz \
        --out_dir analysis_out
"""
import argparse
import os

import numpy as np

DIST_BINS = [0, 15, 30, 50, 80]
NPTS_BINS = [0, 1, 5, 20, 100, 10 ** 9]


def load(path):
    d = dict(np.load(path, allow_pickle=False))
    d['name'] = str(d['name'])
    d['class_names'] = [str(c) for c in d['class_names']]
    d['cls_name'] = np.array([d['class_names'][c - 1] for c in d['cls']])
    d['dist_bin'] = np.array([_bin_label(v, DIST_BINS) for v in d['dist']])
    d['npts_bin'] = np.array([_bin_label(v, NPTS_BINS) for v in d['npts']])
    return d


def _bin_label(v, bins):
    for lo, hi in zip(bins[:-1], bins[1:]):
        if lo <= v < hi:
            return '%d-%d' % (lo, hi) if hi < 10 ** 9 else '>=%d' % lo
    return '>=%d' % bins[-1]


def check_aligned(a, b):
    ok = (len(a['cls']) == len(b['cls']) and np.array_equal(a['frame'], b['frame'])
          and np.array_equal(a['cls'], b['cls']) and np.allclose(a['dist'], b['dist']))
    if not ok:
        raise SystemExit('ERROR: the two feature files do not contain the same objects. Extract both with the same '
                         '--max_frames and --weather.')


def mcnemar(det_a, det_b):
    """Exact McNemar test. Returns (fixed by b, broken by b, two-sided p)."""
    from scipy.stats import binomtest
    fixed = int(np.sum(~det_a & det_b))
    broken = int(np.sum(det_a & ~det_b))
    p = binomtest(min(fixed, broken), fixed + broken, 0.5).pvalue if fixed + broken else 1.0
    return fixed, broken, p


def cluster_bootstrap_ci(det_a, det_b, groups, reps=2000, seed=0):
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    members = [np.flatnonzero(groups == g) for g in uniq]
    deltas = np.empty(reps)
    for r in range(reps):
        idx = np.concatenate([members[k] for k in rng.integers(0, len(uniq), len(uniq))])
        deltas[r] = det_b[idx].mean() - det_a[idx].mean()
    return np.percentile(deltas, [2.5, 97.5])


def holm(pvals):
    order = np.argsort(pvals)
    adj = np.empty(len(pvals))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvals) - rank) * pvals[i]))
        adj[i] = running
    return adj


def subgroup_table(a, b, key):
    rows = []
    for v in sorted(np.unique(a[key])):
        m = a[key] == v
        fixed, broken, p = mcnemar(a['detected'][m], b['detected'][m])
        rows.append(dict(group=key, value=str(v), n=int(m.sum()), recall_a=a['detected'][m].mean(),
                         recall_b=b['detected'][m].mean(), fixed=fixed, broken=broken, p=p))
    if rows:
        for r, padj in zip(rows, holm(np.array([r['p'] for r in rows]))):
            r['delta'] = r['recall_b'] - r['recall_a']
            r['p_holm'] = padj
    return rows


def regressions(a, b, regression_class, lines, out_dir):
    try:
        import pandas as pd
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
    except ImportError:
        lines.append('\n[regression skipped: pip install statsmodels pandas]')
        return

    def frame(d, treat):
        m = d['cls_name'] == regression_class
        return pd.DataFrame(dict(detected=d['detected'][m].astype(int), log2_npts=np.log2(1 + d['npts'][m]),
                                 dist_10m=d['dist'][m] / 10.0, weather=d['weather'][m], light=d['light'][m],
                                 seq=d['sequence'][m], treat=treat))

    df_a, df_b = frame(a, 0), frame(b, 1)
    lines.append('\n=== Why are %s objects missed? Logistic regression, sequence-clustered SEs ===' % regression_class)
    lines.append('Odds ratio > 1 raises the odds of detection. log2_npts: per doubling of (1 + points in box); '
                 'dist_10m: per 10 m farther; weather/light: relative to the first category alphabetically.')
    formula = 'detected ~ log2_npts + dist_10m + C(weather) + C(light)'
    tables = []
    for name, df in [(a['name'], df_a), (b['name'], df_b)]:
        if df['detected'].nunique() < 2:
            lines.append('\n%s: all %s objects have the same outcome; regression not possible' % (name, regression_class))
            continue
        try:
            groups = pd.factorize(df['seq'])[0]
            res = smf.glm(formula, df, family=sm.families.Binomial()).fit(cov_type='cluster',
                                                                         cov_kwds={'groups': groups})
        except Exception as e:  # e.g. perfect separation when a weather has no detections
            lines.append('\n%s: regression failed (%s)' % (name, e))
            continue
        ci = res.conf_int()
        t = pd.DataFrame({'odds_ratio': np.exp(res.params), 'ci_low': np.exp(ci[0]), 'ci_high': np.exp(ci[1]),
                          'p': res.pvalues})
        t.insert(0, 'model', name)
        tables.append(t)
        lines.append('\n%s (n=%d)\n%s' % (name, len(df), t.drop(columns='model').round(4).to_string()))

    both = pd.concat([df_a, df_b], ignore_index=True)
    lines.append('\n=== Where does %s gain over %s? Interaction model (%s) ===' % (b['name'], a['name'], regression_class))
    lines.append('treat = %s. A significant treat:x term means the advantage changes with x '
                 '(odds ratio > 1: the advantage grows as x increases).' % b['name'])
    try:
        groups = pd.factorize(both['seq'])[0]
        res = smf.glm('detected ~ treat * (log2_npts + dist_10m) + C(weather) + C(light)', both,
                      family=sm.families.Binomial()).fit(cov_type='cluster', cov_kwds={'groups': groups})
        ci = res.conf_int()
        t = pd.DataFrame({'odds_ratio': np.exp(res.params), 'ci_low': np.exp(ci[0]), 'ci_high': np.exp(ci[1]),
                          'p': res.pvalues})
        keep = [k for k in t.index if k.startswith('treat')]
        lines.append(t.loc[keep].round(4).to_string())
        t.insert(0, 'model', 'interaction')
        tables.append(t)
    except Exception as e:
        lines.append('interaction model failed (%s)' % e)
    if tables:
        pd.concat(tables).to_csv(os.path.join(out_dir, 'regression_%s_vs_%s.csv' % (a['name'], b['name'])))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--baseline', required=True, help='.npz of the reference model (e.g. radar_scratch)')
    parser.add_argument('--treatment', required=True, help='.npz of the model being tested (e.g. radar_ft_from_livox)')
    parser.add_argument('--out_dir', default='analysis_out')
    parser.add_argument('--regression_class', default='Vehicle',
                        help='class for the regressions (radar barely detects pedestrians/bikes, which makes them '
                             'impossible to model)')
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    a, b = load(args.baseline), load(args.treatment)
    check_aligned(a, b)
    lines = ['%s (baseline) vs %s (treatment), %d labelled objects in %d sequences'
             % (a['name'], b['name'], len(a['cls']), len(np.unique(a['sequence'])))]

    ra, rb = a['detected'].mean(), b['detected'].mean()
    fixed, broken, p = mcnemar(a['detected'], b['detected'])
    lo, hi = cluster_bootstrap_ci(a['detected'], b['detected'], a['sequence'])
    lines += ['\n=== Overall (all classes) ===',
              'recall %s %.4f | %s %.4f | difference %+.4f, 95%% sequence-bootstrap CI [%+.4f, %+.4f]'
              % (a['name'], ra, b['name'], rb, rb - ra, lo, hi),
              'objects fixed by %s: %d, broken: %d, McNemar exact p = %.3g' % (b['name'], fixed, broken, p),
              ('-> the difference is unlikely to be chance on this data' if p < 0.05 and (lo > 0 or hi < 0)
               else '-> not significant on this data')]

    all_rows = []
    for key in ['weather', 'light', 'cls_name', 'dist_bin', 'npts_bin']:
        rows = subgroup_table(a, b, key)
        all_rows += rows
        lines.append('\n=== By %s (McNemar, Holm-corrected within this grouping) ===' % key)
        lines.append('%-14s %7s %10s %10s %9s %7s %7s %10s %10s' % ('value', 'n', 'recall_a', 'recall_b', 'delta',
                                                                 'fixed', 'broken', 'p', 'p_holm'))
        for r in rows:
            flag = '  *' if r['p_holm'] < 0.05 else ''
            lines.append('%-14s %7d %10.4f %10.4f %+9.4f %7d %7d %10.3g %10.3g%s'
                         % (r['value'], r['n'], r['recall_a'], r['recall_b'], r['delta'], r['fixed'], r['broken'],
                            r['p'], r['p_holm'], flag))
    lines.append('\n* = significant after Holm correction (p_holm < 0.05)')

    import csv
    with open(os.path.join(args.out_dir, 'subgroups_%s_vs_%s.csv' % (a['name'], b['name'])), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['group', 'value', 'n', 'recall_a', 'recall_b', 'delta', 'fixed', 'broken',
                                          'p', 'p_holm'])
        w.writeheader()
        w.writerows(all_rows)

    regressions(a, b, args.regression_class, lines, args.out_dir)
    lines.append('\nNote: these tests cover variation in the validation data, not training randomness; repeated '
                 'training seeds are needed to generalise beyond these two trained models.')
    text = '\n'.join(lines)
    print(text)
    with open(os.path.join(args.out_dir, 'stats_%s_vs_%s.txt' % (a['name'], b['name'])), 'w') as f:
        f.write(text + '\n')


if __name__ == '__main__':
    main()
