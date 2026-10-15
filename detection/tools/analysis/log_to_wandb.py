"""
Upload an analysis folder to Weights & Biases as one run: every PNG as an image, every CSV as a table, and the whole
folder (feature .npz files, stats text) as an artifact. Falls back to offline mode without internet.

    python analysis/log_to_wandb.py --dir ~/analysis/20261015 --project dsert-single-sensor --name analysis_20261015
"""
import argparse
import glob
import os


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dir', required=True)
    parser.add_argument('--project', required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--entity', default=None)
    args = parser.parse_args()

    import pandas as pd
    import wandb

    root = os.path.abspath(args.dir)
    kwargs = dict(project=args.project, entity=args.entity, name=args.name, job_type='analysis', dir=root,
                  config={'analysis_dir': root})
    try:
        run = wandb.init(**kwargs, settings=wandb.Settings(init_timeout=120))
    except Exception as e:
        print('wandb unreachable (%r); logging offline' % (e,))
        run = wandb.init(**kwargs, mode='offline')

    logged = {}
    for png in sorted(glob.glob(os.path.join(root, '**', '*.png'), recursive=True)):
        key = 'figures/' + os.path.relpath(png, root).replace(os.sep, '/')[:-4]
        logged[key] = wandb.Image(png)
    for csv in sorted(glob.glob(os.path.join(root, '**', '*.csv'), recursive=True)):
        key = 'tables/' + os.path.relpath(csv, root).replace(os.sep, '/')[:-4]
        logged[key] = wandb.Table(dataframe=pd.read_csv(csv))
    run.log(logged)

    artifact = wandb.Artifact(args.name, type='analysis')
    for path in glob.glob(os.path.join(root, '**', '*'), recursive=True):
        if os.path.isfile(path) and '/wandb/' not in path.replace(os.sep, '/'):
            artifact.add_file(path, name=os.path.relpath(path, root))
    run.log_artifact(artifact)
    print('logged %d figures/tables + artifact to %s' % (len(logged), getattr(run, 'url', None) or 'offline run'))
    run.finish()


if __name__ == '__main__':
    main()
