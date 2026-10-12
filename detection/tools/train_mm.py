import os
import glob
import argparse
import datetime
from pathlib import Path


import torch
import torch.distributed as dist
import torch.nn as nn
from tensorboardX import SummaryWriter

from al3d_utils import common_utils
from al3d_utils.config_utils import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from al3d_utils.model_utils import load_params_from_file, load_params_with_optimizer
from al3d_utils.optimize_utils import build_optimizer, build_scheduler
from al3d_utils.train_utils import train_model

from al3d_det.datasets import build_dataloader
from al3d_det.models import build_network, model_fn_decorator
from test import repeat_eval_ckpt

import pdb

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')

    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=None, required=False, help='number of epochs to train for')
    parser.add_argument('--workers', type=int, default=8, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained_model')
    parser.add_argument('--fix_pretrained_weights', action='store_true', default=False, help='')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--sync_bn', action='store_true', default=False, help='whether to use sync bn')
    parser.add_argument('--fix_random_seed', action='store_true', default=False, help='')
    parser.add_argument('--ckpt_save_interval', type=int, default=1, help='number of training epochs')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--max_ckpt_save_num', type=int, default=30, help='max number of saved checkpoint')
    parser.add_argument('--merge_all_iters_to_one_epoch', action='store_true', default=False, help='')
    parser.add_argument('--find_unused_parameters', action='store_true', default=False, help='')             
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')
    parser.add_argument('--max_waiting_mins', type=int, default=0, help='max waiting minutes')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')
    parser.add_argument('--amp', choices=['none', 'fp16', 'bf16'], default='none',
                        help='mixed precision for the encoder (models with set_amp, e.g. CenterPointSingle)')
    parser.add_argument('--compile', action='store_true', default=False,
                        help='torch.compile the dense layers (models with compile_dense, e.g. CenterPointSingle)')
    parser.add_argument('--compile_mode', type=str, default='default',
                        help='torch.compile mode: default | reduce-overhead | max-autotune')
    parser.add_argument('--pretrained_scope', choices=['all', 'encoder'], default='all',
                        help="with --pretrained_model: 'encoder' copies only the 3D/BEV backbones, heads start fresh")
    parser.add_argument('--skip_final_eval', action='store_true', default=False,
                        help='do not run the evaluation that follows training')

    args = parser.parse_args()
    # torchrun passes the local rank through the environment rather than --local_rank.
    args.local_rank = int(os.environ.get('LOCAL_RANK', args.local_rank))

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.ROOT_DIR = (Path(__file__).resolve().parent / '../').resolve()
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg


ENCODER_PREFIXES = ('vfe.', 'backbone3d.', 'map_to_bev.', 'backbone2d.')


def load_pretrained(model, filename, logger, fix_pretrained_weights=False, scope='all'):
    """
    Load matching weights from a checkpoint. Strips 'module.' (DDP) and 'lidar.' (CenterPointMM) prefixes so
    a CenterPointSingle checkpoint, a CenterPointMM checkpoint, and the LiDAR branch of CenterPointMM all
    load into one another. Raises if nothing matches, since a silent no-op load would invalidate a
    pretraining comparison.
    """
    checkpoint = torch.load(filename, map_location='cpu')
    disk_state = checkpoint.get('model_state', checkpoint)
    own_state = model.state_dict()

    def strip(key):
        for prefix in ('module.', 'lidar.'):
            if key.startswith(prefix):
                key = key[len(prefix):]
        return key

    loaded, skipped_shape = {}, []
    for key, val in disk_state.items():
        k = key if key in own_state else strip(key)
        if k not in own_state:
            continue
        if scope == 'encoder' and not k.startswith(ENCODER_PREFIXES):
            continue
        if own_state[k].shape != val.shape:
            skipped_shape.append(k)
            continue
        loaded[k] = val
    if not loaded:
        raise RuntimeError('No parameter of %s matches the model (checkpoint keys look like %s)'
                           % (filename, list(disk_state)[:3]))
    own_state.update(loaded)
    model.load_state_dict(own_state)
    if fix_pretrained_weights:
        for name, param in model.named_parameters():
            if name in loaded:
                param.requires_grad = False
    missing = [k for k in own_state if k not in loaded]
    logger.info('==> Pretrained %s (scope=%s): loaded %d/%d tensors, %d shape mismatches, %d left at init'
                % (filename, scope, len(loaded), len(own_state), len(skipped_shape), len(missing)))
    for k in skipped_shape:
        logger.info('    shape mismatch: %s' % k)
    for k in missing:
        logger.info('    left at init: %s' % k)


def main():
    args, cfg = parse_config()
    # TF32 matmuls/convs: large speedup for the fp32 parts on Ampere/Hopper at negligible accuracy cost.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    if args.launcher == 'none':
        dist_train = False
        total_gpus = 1
    else:
        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl'
        )
        dist_train = True

    if args.batch_size is None:
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    else:
        assert args.batch_size % total_gpus == 0, 'Batch size should match the number of gpus'
        args.batch_size = args.batch_size // total_gpus
    args.epochs = cfg.OPTIMIZATION.NUM_EPOCHS if args.epochs is None else args.epochs

    if args.fix_random_seed:
        common_utils.set_random_seed(666)

    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    ckpt_dir = output_dir / 'ckpt'
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / ('log_train_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    # log to file
    logger.info('**********************Start logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    if dist_train:
        logger.info('total_batch_size: %d' % (total_gpus * args.batch_size))
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger)
    if cfg.LOCAL_RANK == 0:
        os.system('cp %s %s' % (args.cfg_file, output_dir))

    tb_log = SummaryWriter(log_dir=str(output_dir / 'tensorboard')) if cfg.LOCAL_RANK == 0 else None

    # -----------------------create dataloader & network & optimizer---------------------------
    train_set, train_loader, train_sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_train,
        workers=args.workers,
        logger=logger,
        training=True,
        merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,
        total_epochs=args.epochs
    )
    
    
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=train_set)
    if args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()

    amp_dtype = {'none': None, 'fp16': torch.float16, 'bf16': torch.bfloat16}[args.amp]
    if amp_dtype is not None:
        assert hasattr(model, 'set_amp'), '--amp is only wired into models with set_amp (CenterPointSingle)'
        model.set_amp(amp_dtype)
    # bf16 has fp32's exponent range and needs no loss scaling; fp16 does.
    scaler = torch.cuda.amp.GradScaler() if args.amp == 'fp16' else None
    logger.info('Mixed precision: %s%s' % (args.amp, ' (with GradScaler)' if scaler is not None else ''))

    optimizer = build_optimizer(model, cfg.OPTIMIZATION)
    # load checkpoint if it is possible
    
    start_epoch = it = 0
    last_epoch = -1
    
    if args.pretrained_model is not None:
        load_pretrained(getattr(model, 'lidar', model), args.pretrained_model, logger,
                        fix_pretrained_weights=args.fix_pretrained_weights, scope=args.pretrained_scope)

    if args.ckpt is not None:
        # it, start_epoch = load_params_with_optimizer(model.lidar, args.ckpt, to_cpu=dist, optimizer=optimizer, logger=logger)
        it, start_epoch = load_params_with_optimizer(model, args.ckpt, to_cpu=dist, optimizer=optimizer, logger=logger)
        last_epoch = start_epoch - 1
    else:
        # ckpt_list = glob.glob(str(ckpt_dir / '*checkpoint_epoch_*.pth'))
        # if len(ckpt_list) > 0:
        #     ckpt_list.sort(key=os.path.getmtime)
        #     it, start_epoch = load_params_with_optimizer(
        #         model.lidar, ckpt_list[-1], to_cpu=dist, optimizer=optimizer, logger=logger
        #     )
        #     last_epoch = start_epoch - 1
        
        ckpt_list = glob.glob(str(ckpt_dir / '*checkpoint_epoch_*.pth'))
        if len(ckpt_list) > 0:
            ckpt_list.sort(key=os.path.getmtime)
            it, start_epoch = load_params_with_optimizer(
                model, ckpt_list[-1], to_cpu=dist, optimizer=optimizer, logger=logger
            )
            last_epoch = start_epoch - 1
        
    
    # last_epoch = 0
    model.train()  # before wrap to DistributedDataParallel to support fixed some parameters

    if args.compile:
        assert hasattr(model, 'compile_dense'), '--compile is only wired into models with compile_dense (CenterPointSingle)'
        import torch._dynamo
        # Box/ROI counts change between iterations; give dynamo room for a few shape variants.
        torch._dynamo.config.cache_size_limit = 64
        n = model.compile_dense(mode=args.compile_mode)
        logger.info('torch.compile (%s) applied to %d dense submodules' % (args.compile_mode, n))
    
    if dist_train:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[cfg.LOCAL_RANK % torch.cuda.device_count()], find_unused_parameters=args.find_unused_parameters)
        ### multimodal config
        
    logger.info(model)

    lr_scheduler, lr_warmup_scheduler = build_scheduler(
        optimizer, total_iters_each_epoch=len(train_loader), total_epochs=args.epochs,
        last_epoch=last_epoch, optim_cfg=cfg.OPTIMIZATION
    )

    # -----------------------start training---------------------------
    logger.info('**********************Start training %s/%s(%s)**********************'
                % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))
    train_model(
        model,
        optimizer,
        train_loader,
        model_func=model_fn_decorator(),
        lr_scheduler=lr_scheduler,
        optim_cfg=cfg.OPTIMIZATION,
        start_epoch=start_epoch,
        total_epochs=args.epochs,
        start_iter=it,
        rank=cfg.LOCAL_RANK,
        tb_log=tb_log,
        ckpt_save_dir=ckpt_dir,
        train_sampler=train_sampler,
        lr_warmup_scheduler=lr_warmup_scheduler,
        ckpt_save_interval=args.ckpt_save_interval,
        max_ckpt_save_num=args.max_ckpt_save_num,
        merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,
        scaler=scaler
    )

    logger.info('**********************End training %s/%s(%s)**********************\n\n\n'
                % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))

    if args.skip_final_eval:
        return
    logger.info('**********************Start evaluation %s/%s(%s)**********************' %
                (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))
    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_train, workers=args.workers, logger=logger, training=False
    )
    eval_output_dir = output_dir / 'eval' / 'eval_with_train'
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    args.start_epoch = max(args.epochs, 0)  # Only evaluate the last 3 epochs

    repeat_eval_ckpt(
        model.module if dist_train else model,
        test_loader, args, eval_output_dir, logger, ckpt_dir,
        dist_test=dist_train
    )
    logger.info('**********************End evaluation %s/%s(%s)**********************' %
                (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))


if __name__ == '__main__':
    main()
