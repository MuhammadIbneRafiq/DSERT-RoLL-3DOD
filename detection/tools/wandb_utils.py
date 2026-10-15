"""
Optional Weights & Biases logging. It mirrors the scalars the code already writes to TensorBoard, so the
training loop needs no changes: train/* and meta_data/* use the iteration as x-axis, eval/* uses the epoch.

Enabled with `train_mm.py --wandb_project <name>`. Without that flag, or without wandb installed, nothing here runs.
"""
import numbers
import secrets
from pathlib import Path

import torch

try:
    import wandb
except ImportError:
    wandb = None


def active():
    return wandb is not None and wandb.run is not None


def flatten_scalars(d, prefix=''):
    """{'overall': {'Vehicle_AP': 0.5}, 'counts': {...}} -> {'overall/Vehicle_AP': 0.5, ...}, numbers only."""
    out = {}
    for key, val in d.items():
        name = '%s/%s' % (prefix, key) if prefix else str(key)
        if isinstance(val, dict):
            out.update(flatten_scalars(val, name))
        elif torch.is_tensor(val) and val.numel() == 1:
            out[name] = val.item()
        elif isinstance(val, numbers.Number) and not isinstance(val, bool):
            out[name] = float(val)
    return out


def _plain(obj):
    """Config/args -> JSON-friendly values for wandb.config (EasyDict, Path, numpy are not)."""
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    return str(obj)


def init(args, cfg, output_dir, logger, job_type='train'):
    """Start (or resume) the run for this output dir. Call on rank 0 before training or evaluation."""
    if not getattr(args, 'wandb_project', None):
        return None
    if wandb is None:
        raise ImportError('--wandb_project was given but wandb is not installed: pip install wandb')

    # One wandb run per output dir: a resubmitted Slurm job with the same --extra_tag continues the same run.
    id_file = Path(output_dir) / 'wandb_run_id.txt'
    run_id = id_file.read_text().strip() if id_file.exists() else secrets.token_hex(4)
    id_file.write_text(run_id)

    kwargs = dict(
        project=args.wandb_project,
        entity=getattr(args, 'wandb_entity', None),
        name='%s/%s' % (cfg.TAG, args.extra_tag),
        group=cfg.TAG,
        job_type=job_type,
        tags=[str(t) for t in [cfg.DATA_CONFIG.get('SENSOR', {}).get('NAME'), job_type,
                               'amp-' + args.amp if getattr(args, 'amp', None) else None] if t],
        id=run_id,
        resume='allow',
        dir=str(output_dir),
        config={'cfg': _plain(cfg), 'args': _plain(vars(args))},
    )
    try:
        run = wandb.init(**kwargs, settings=wandb.Settings(init_timeout=120))
    except Exception as e:  # no internet on the node, wandb outage, ...: keep training, log locally
        logger.warning('wandb could not reach the server (%r). Logging offline; upload afterwards with:\n'
                       '    wandb sync %s/wandb/offline-run-*' % (e, output_dir))
        run = wandb.init(**kwargs, mode='offline')

    wandb.define_metric('epoch')
    wandb.define_metric('eval/*', step_metric='epoch')
    logger.info('wandb run: %s (%s)' % (run.name, getattr(run, 'url', None) or 'offline'))
    return run


class MirroredWriter:
    """Wraps a tensorboardX SummaryWriter; every add_scalar also goes to wandb at the same step."""

    def __init__(self, writer):
        self.writer = writer

    def add_scalar(self, tag, value, global_step=None, *args, **kwargs):
        self.writer.add_scalar(tag, value, global_step, *args, **kwargs)
        if torch.is_tensor(value):
            value = value.item()
        wandb.log({tag: float(value)}, step=int(global_step) if global_step is not None else None)

    def __getattr__(self, name):
        return getattr(self.writer, name)


def log_eval(metrics, epoch):
    """Log a (possibly nested) evaluation result dict under eval/*, against the epoch."""
    if not active():
        return
    flat = {'eval/' + k: v for k, v in flatten_scalars(metrics).items()}
    flat['epoch'] = int(float(epoch))
    wandb.log(flat)


def finish():
    if active():
        wandb.finish()
