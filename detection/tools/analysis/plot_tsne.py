"""
Compare encoders from the .npz files written by extract_features.py. CPU only (numpy, scikit-learn, matplotlib).

Per model it writes a t-SNE figure (coloured by class / weather / detected / distance) and computes numbers that,
unlike t-SNE pictures, can be compared between models:
  * knn_class_acc   - 10-NN accuracy of predicting the object class from the feature (5-fold CV). Higher = classes
                      are better separated in the encoder's feature space.
  * silhouette_class - how tightly features cluster by class (-1..1, higher = cleaner clusters).
  * knn_weather_acc - how well the weather can be predicted from object features. Higher = the encoder's view of an
                      object changes more with the weather (less weather-invariant).
  * recall tables   - fraction of objects detected, per weather / distance / points-in-box. This is where a model
                      "breaks".
t-SNE axes mean nothing across models; compare the numbers, and use the pictures for intuition.

    python analysis/plot_tsne.py --features feats_a.npz feats_b.npz --out_dir analysis_out
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
    return d


def balanced_subset(cls, cap, seed=0):
    """Indices with at most `cap` objects per class, so vehicles do not dominate the plots and metrics."""
    rng = np.random.default_rng(seed)
    idx = []
    for c in np.unique(cls):
        members = np.flatnonzero(cls == c)
        idx.append(rng.choice(members, size=min(cap, len(members)), replace=False))
    return np.sort(np.concatenate(idx))


def embed_space(feat):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    x = StandardScaler().fit_transform(feat)
    return PCA(n_components=min(50, x.shape[1], len(x) - 1), random_state=0).fit_transform(x)


def metrics(d, idx):
    from sklearn.metrics import silhouette_score
    from sklearn.model_selection import cross_val_score
    from sklearn.neighbors import KNeighborsClassifier
    x = embed_space(d['feat'][idx])
    knn = KNeighborsClassifier(n_neighbors=10)
    out = {
        'objects': len(idx),
        'knn_class_acc': cross_val_score(knn, x, d['cls'][idx], cv=5).mean(),
        'silhouette_class': silhouette_score(x, d['cls'][idx], metric='cosine'),
    }
    weathers = d['weather'][idx]
    if len(np.unique(weathers)) > 1:
        out['knn_weather_acc'] = cross_val_score(knn, x, weathers, cv=5).mean()
    out['recall_all'] = d['detected'].mean()
    return out, x


def recall_table(d, key, bins=None):
    rows = []
    if bins is None:
        for v in sorted(np.unique(d[key])):
            m = d[key] == v
            rows.append((str(v), m.sum(), d['detected'][m].mean()))
    else:
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (d[key] >= lo) & (d[key] < hi)
            label = '%d-%d' % (lo, hi) if hi < 10 ** 9 else '>=%d' % lo
            rows.append((label, m.sum(), d['detected'][m].mean() if m.any() else np.nan))
    return rows


def plot_tsne(d, x, idx, out_png):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    emb = TSNE(n_components=2, perplexity=30, init='pca', learning_rate='auto', random_state=0).fit_transform(x)
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    panels = [
        ('class', [d['class_names'][c - 1] for c in d['cls'][idx]]),
        ('weather', list(d['weather'][idx])),
        ('detected', ['detected' if v else 'missed' for v in d['detected'][idx]]),
    ]
    fixed_colors = {'detected': 'tab:green', 'missed': 'tab:red'}
    for ax, (title, labels) in zip(axes, panels):
        labels = np.array(labels)
        for lab in sorted(np.unique(labels)):
            m = labels == lab
            ax.scatter(emb[m, 0], emb[m, 1], s=3, label='%s (%d)' % (lab, m.sum()), alpha=0.7,
                       c=fixed_colors.get(lab))
        ax.set_title('%s: %s' % (d['name'], title))
        ax.legend(markerscale=4, fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    sc = axes[3].scatter(emb[:, 0], emb[:, 1], s=3, c=d['dist'][idx], cmap='viridis')
    axes[3].set_title('%s: distance (m)' % d['name'])
    axes[3].set_xticks([]); axes[3].set_yticks([])
    fig.colorbar(sc, ax=axes[3])
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def plot_recall_comparison(models, out_png):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    for ax, (key, bins, title) in zip(axes, [('weather', None, 'recall by weather'),
                                              ('dist', DIST_BINS, 'recall by distance (m)'),
                                              ('npts', NPTS_BINS, 'recall by points inside the box')]):
        width = 0.8 / len(models)
        for k, d in enumerate(models):
            rows = recall_table(d, key, bins)
            pos = np.arange(len(rows)) + k * width
            ax.bar(pos, [r[2] for r in rows], width=width, label=d['name'])
            ax.set_xticks(np.arange(len(rows)) + 0.4 - width / 2)
            ax.set_xticklabels([r[0] for r in rows], rotation=20)
        ax.set_ylim(0, 1)
        ax.set_title(title)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--features', nargs='+', required=True, help='.npz files from extract_features.py')
    parser.add_argument('--out_dir', default='analysis_out')
    parser.add_argument('--per_class_cap', type=int, default=1500, help='max objects per class in t-SNE/metrics')
    parser.add_argument('--no_tsne', action='store_true', help='only metrics and recall tables (fast)')
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    models = [load(p) for p in args.features]
    summary = []
    for d in models:
        idx = balanced_subset(d['cls'], args.per_class_cap)
        m, x = metrics(d, idx)
        summary.append((d['name'], m))
        if not args.no_tsne:
            png = os.path.join(args.out_dir, 'tsne_%s.png' % d['name'])
            plot_tsne(d, x, idx, png)
            print('wrote', png)
    plot_recall_comparison(models, os.path.join(args.out_dir, 'recall_comparison.png'))

    keys = ['objects', 'knn_class_acc', 'silhouette_class', 'knn_weather_acc', 'recall_all']
    lines = ['model,' + ','.join(keys)]
    print('\n%-22s' % 'model' + ''.join('%18s' % k for k in keys))
    for name, m in summary:
        vals = [m.get(k, np.nan) for k in keys]
        print('%-22s' % name + ''.join('%18.3f' % v if k != 'objects' else '%18d' % v for k, v in zip(keys, vals)))
        lines.append(name + ',' + ','.join('%.4f' % v for v in vals))
    with open(os.path.join(args.out_dir, 'metrics.csv'), 'w') as f:
        f.write('\n'.join(lines) + '\n')

    for key, bins, title in [('weather', None, 'weather'), ('dist', DIST_BINS, 'distance (m)'),
                             ('npts', NPTS_BINS, 'points in box')]:
        print('\nrecall by %s' % title)
        tables = [recall_table(d, key, bins) for d in models]
        print('%-12s' % title[:12] + ''.join('%20s' % d['name'][:20] for d in models) + '%10s' % 'objects')
        for r, row in enumerate(tables[0]):
            print('%-12s' % row[0] + ''.join('%20.3f' % t[r][2] for t in tables) + '%10d' % row[1])
    print('\nfigures and metrics.csv in', os.path.abspath(args.out_dir))


if __name__ == '__main__':
    main()
