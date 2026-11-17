#!/usr/bin/env python3
"""Full HAMMER training with torchrun. Place beside the official train.py.

Uses the official dataset/transforms, six losses, optimizer and training loop.
No CRAF, mixed precision, or gradient accumulation is introduced.
"""
import argparse
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import random
import sys
import time


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo-dir', help='Official HAMMER repository root (contains train.py and models/HAMMER.py)')
    p.add_argument('--config', default='configs/train.yaml')
    p.add_argument('--data-root', required=True, help='Parent containing DGM4/')
    p.add_argument('--train-file')
    p.add_argument('--val-file')
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--checkpoint', help='Initial ALBEF or HAMMER weights')
    group.add_argument('--resume', help='checkpoint_last.pth from this launcher')
    p.add_argument('--output-dir', required=True)
    p.add_argument('--text-encoder', default='bert-base-uncased')
    p.add_argument('--batch-size', type=int, default=32, help='Per GPU')
    p.add_argument('--val-batch-size', type=int, default=32)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--token-momentum', action=argparse.BooleanOptionalAction, default=True,
                   help='Token momentum teacher; enabled as in the official train.sh')
    p.add_argument('--check-only', action='store_true')
    return p.parse_args()


def resolve_repo(repo_dir=None):
    required = ('train.py', 'models/HAMMER.py')
    candidates = ([Path(repo_dir).expanduser().resolve()] if repo_dir else
                  list(dict.fromkeys([Path.cwd().resolve(), Path(__file__).resolve().parent,
                                      Path(__file__).resolve().parent.parent])))
    for candidate in candidates:
        if all((candidate / name).is_file() for name in required):
            return candidate
    details = '\n'.join(f'  {candidate}: missing ' + ', '.join(
        name for name in required if not (candidate / name).is_file()) for candidate in candidates)
    raise ValueError('Official HAMMER source repository was not found.\n' + details +
                     '\nThe launcher alone is not the model source. Clone/copy the full official '
                     'repository and pass --repo-dir /absolute/path/to/repository.')


def load_training_config(path):
    """Coerce numeric fields that PyYAML may parse as strings (e.g. 2e-5)."""
    import yaml
    with Path(path).open() as f:
        cfg = yaml.safe_load(f)
    numeric_fields = {
        'optimizer': ('lr', 'lr_img', 'lr_text', 'weight_decay', 'momentum', 'opt_eps'),
        'schedular': ('lr', 'min_lr', 'warmup_lr', 'decay_rate', 'lr_cycle_mul',
                      'lr_noise_pct', 'lr_noise_std'),
    }
    for section, fields in numeric_fields.items():
        for field in fields:
            value = cfg.get(section, {}).get(field)
            if value is None:
                continue
            try:
                if isinstance(value, bool):
                    raise ValueError('boolean is not a numeric hyperparameter')
                number = float(value)
                if not math.isfinite(number):
                    raise ValueError('non-finite hyperparameter')
            except (TypeError, ValueError) as error:
                raise ValueError(f'{section}.{field} must be a finite number, got {value!r}') from error
            cfg[section][field] = number
    return cfg


def prepare(a):
    root = resolve_repo(a.repo_dir)
    a.repo_dir = str(root)
    sys.path.insert(0, str(root))
    cfg_path = Path(a.config).expanduser()
    if not cfg_path.is_absolute():
        cfg_path = root / cfg_path
    cfg_path = cfg_path.resolve()
    cfg = load_training_config(cfg_path)
    a.data_root = str(Path(a.data_root).resolve())
    a.output_dir = str(Path(a.output_dir).resolve())
    cfg['train_file'] = [str(Path(a.train_file or Path(a.data_root) / 'DGM4/metadata/train.json').resolve())]
    cfg['val_file'] = [str(Path(a.val_file or Path(a.data_root) / 'DGM4/metadata/val.json').resolve())]
    bert_config = Path(cfg['bert_config'])
    cfg['bert_config'] = str((bert_config if bert_config.is_absolute() else root / bert_config).resolve())
    cfg['batch_size_train'] = a.batch_size
    cfg['batch_size_val'] = a.val_batch_size
    cfg['schedular']['epochs'] = a.epochs
    if min(a.batch_size, a.val_batch_size, a.epochs) < 1 or a.workers < 0:
        raise ValueError('Batch sizes/epochs must be positive; workers must be nonnegative.')
    for path in [cfg['bert_config'], a.checkpoint or a.resume]:
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    counts = []
    for split in ['train_file', 'val_file']:
        with open(cfg[split][0]) as f:
            rows = json.load(f)
        if not isinstance(rows, list) or not rows:
            raise ValueError(f'{split}: expected a nonempty JSON list')
        required = {'image', 'text', 'fake_cls', 'fake_image_box', 'fake_text_pos'}
        if not required.issubset(rows[0]):
            raise ValueError(f'{split}: missing annotation fields {required - rows[0].keys()}')
        # Metadata checks only: image files are read by the original dataset during training.
        counts.append(len(rows))
    world = int(os.environ.get('WORLD_SIZE', '4'))
    global_batch = world * a.batch_size
    if cfg['queue_size'] % global_batch:
        raise ValueError(f'queue_size={cfg["queue_size"]} must be divisible by global batch={global_batch}')
    steps = math.ceil(counts[0] / world) // a.batch_size
    if not steps:
        raise ValueError('Training data is smaller than one distributed batch.')
    if int(os.environ.get('RANK', '0')) == 0:
        print(json.dumps({'train_samples': counts[0], 'val_samples': counts[1],
                          'world_size': world, 'per_gpu_batch': a.batch_size,
                          'global_batch': global_batch, 'steps_per_epoch': steps,
                          'epochs': a.epochs, 'total_steps': steps * a.epochs,
                          'data_root': a.data_root,
                          'image_files_checked': False}, indent=2), flush=True)
    return cfg


def install_import_compatibility():
    # Official source expects old ruamel_yaml and a transformers decorator keyword.
    import yaml
    sys.modules.setdefault('ruamel_yaml', yaml)
    import transformers.file_utils as file_utils
    original = file_utils.add_code_sample_docstrings

    def compatible(*args, **kwargs):
        kwargs.pop('tokenizer_class', None)
        return original(*args, **kwargs)

    file_utils.add_code_sample_docstrings = compatible


def atomic_save(torch, state, path):
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(state, temporary)
    os.replace(temporary, path)


METRICS = ['AUC_cls', 'ACC_cls', 'EER_cls', 'MAP', 'OP', 'OR', 'OF1',
           'CP', 'CR', 'CF1', 'OP_k', 'OR_k', 'OF1_k', 'CP_k', 'CR_k', 'CF1_k',
           'IOU_score', 'IOU_ACC_50', 'IOU_ACC_75', 'IOU_ACC_95',
           'ACC_tok', 'Precision_tok', 'Recall_tok', 'F1_tok']


def run(a, cfg):
    import numpy as np
    import torch
    import torch.distributed as dist
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required. Use --check-only for CPU path/metadata checks.')
    if not all(k in os.environ for k in ['RANK', 'WORLD_SIZE', 'LOCAL_RANK']):
        raise RuntimeError('Launch with torchrun --standalone --nproc_per_node=4')
    rank, world, local = (int(os.environ[k]) for k in ['RANK', 'WORLD_SIZE', 'LOCAL_RANK'])
    if world != 4:
        raise ValueError(f'This launcher requires exactly 4 processes, received {world}.')
    torch.cuda.set_device(local)
    # Rank 0 evaluates the whole validation set while other ranks wait.
    dist.init_process_group('nccl', init_method='env://', timeout=timedelta(hours=4))
    writer = None
    try:
        install_import_compatibility()
        import train as upstream
        from dataset import create_dataset, create_loader
        from models.vit import interpolate_pos_embed
        from torch.utils.data.distributed import DistributedSampler
        from torch.utils.tensorboard import SummaryWriter
        from transformers import BertTokenizerFast
        import yaml

        a.rank, a.world_size, a.gpu = rank, world, local
        a.distributed, a.log = True, rank == 0
        a.device, a.lr = 'cuda', cfg['optimizer']['lr']
        device = torch.device('cuda', local)
        torch.manual_seed(a.seed + rank)
        np.random.seed(a.seed + rank)
        random.seed(a.seed + rank)
        torch.backends.cudnn.benchmark = True
        output = Path(a.output_dir)
        if rank == 0:
            output.mkdir(parents=True, exist_ok=True)
            if not a.resume and any((output / name).exists() for name in ['checkpoint_last.pth', 'log.jsonl']):
                raise ValueError('Output contains an existing run. Use a new directory or --resume.')
            with (output / 'config.yaml').open('w') as f:
                yaml.safe_dump(cfg, f)
            with (output / 'run_args.json').open('w') as f:
                json.dump(vars(a), f, indent=2)
            writer = SummaryWriter(str(output / 'tensorboard'))
        dist.barrier()
        train_set, val_set = create_dataset(cfg)
        train_set.root_dir = a.data_root
        val_set.root_dir = a.data_root
        sampler = DistributedSampler(train_set, num_replicas=world, rank=rank,
                                     shuffle=True, seed=a.seed)
        train_loader, val_loader = create_loader(
            [train_set, val_set], [sampler, None], [a.batch_size, a.val_batch_size],
            [a.workers, a.workers], [True, False], [None, None])
        tokenizer = BertTokenizerFast.from_pretrained(a.text_encoder)
        # All encoder weights come from the provided checkpoint. Avoid unnecessary
        # DeiT/BERT model downloads; tokenizer files are still needed.
        from models.xbert import BertForTokenClassification
        old_factory = BertForTokenClassification.__dict__['from_pretrained'] if 'from_pretrained' in BertForTokenClassification.__dict__ else None

        def from_config(cls, *args, **kwargs):
            return cls(config=kwargs['config'], label_smoothing=kwargs.get('label_smoothing', 0.0))

        BertForTokenClassification.from_pretrained = classmethod(from_config)
        try:
            model = upstream.HAMMER(args=a, config=cfg, text_encoder=a.text_encoder,
                                    tokenizer=tokenizer, init_deit=False)
        finally:
            if old_factory is None:
                del BertForTokenClassification.from_pretrained
            else:
                BertForTokenClassification.from_pretrained = old_factory
        # Explicit safe loading: no arbitrary pickle objects from checkpoint files.
        checkpoint = torch.load(a.resume or a.checkpoint, map_location='cpu', weights_only=True)
        state = checkpoint['model']
        if a.resume:
            model.load_state_dict(state, strict=True)
        else:
            state = dict(state)
            for name, encoder in [('visual_encoder', model.visual_encoder),
                                  ('visual_encoder_m', model.visual_encoder_m)]:
                key = name + '.pos_embed'
                if key in state:
                    state[key] = interpolate_pos_embed(state[key], encoder)
            incompatible = model.load_state_dict(state, strict=False)
            # ALBEF lacks HAMMER-specific task heads; encoders must be complete.
            missing_encoder = [k for k in incompatible.missing_keys if k.startswith(
                ('visual_encoder.', 'visual_encoder_m.', 'text_encoder.bert.', 'text_encoder_m.bert.'))]
            if missing_encoder:
                raise ValueError(f'Checkpoint missing encoder weights: {missing_encoder[:20]}')
            if rank == 0:
                print('Checkpoint missing/unexpected keys:', incompatible, flush=True)
        model.to(device)
        optimizer = upstream.create_optimizer(upstream.utils.AttrDict(cfg['optimizer']), model)
        scheduler, _ = upstream.create_scheduler(upstream.utils.AttrDict(cfg['schedular']), optimizer)
        start_epoch, best = 0, -float('inf')
        if a.resume:
            if checkpoint.get('launcher') != 'hammer_torchrun_v1':
                raise ValueError('Resume requires a checkpoint produced by this launcher. Use --checkpoint for weights only.')
            optimizer.load_state_dict(checkpoint['optimizer'])
            if 'lr_scheduler' in checkpoint:
                scheduler.load_state_dict(checkpoint['lr_scheduler'])
            start_epoch = int(checkpoint['epoch']) + 1
            best = float(checkpoint['best_auc'])
        del checkpoint, state
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local])
        plain = model.module
        warmup = cfg['schedular']['warmup_epochs']
        for epoch in range(start_epoch, a.epochs):
            started = time.time()
            train_stats = upstream.train(a, model, train_loader, optimizer, tokenizer,
                                         epoch, warmup, device, scheduler, cfg, writer)
            # No DDP forward in evaluation: peers must not perform broadcast/all-gather.
            metrics_tensor = torch.zeros(len(METRICS), dtype=torch.float64, device=device)
            if rank == 0:
                values = upstream.evaluation(a, plain, val_loader, tokenizer, device, cfg)
                if len(values) != len(METRICS):
                    raise RuntimeError('Unexpected upstream evaluation result length')
                metrics_tensor.copy_(torch.tensor(values, dtype=torch.float64, device=device))
            dist.broadcast(metrics_tensor, src=0)
            metrics = dict(zip(METRICS, metrics_tensor.cpu().tolist()))
            improved = metrics['AUC_cls'] > best
            best = max(best, metrics['AUC_cls'])
            # Save the LR state for the NEXT epoch, including for resume.
            if cfg['schedular']['sched'] != 'cosine_in_step':
                scheduler.step(epoch + warmup + 1)
            if rank == 0:
                state = {'launcher': 'hammer_torchrun_v1', 'model': plain.state_dict(),
                         'optimizer': optimizer.state_dict(), 'config': cfg,
                         'epoch': epoch, 'best_auc': best}
                if cfg['schedular']['sched'] != 'cosine_in_step':
                    state['lr_scheduler'] = scheduler.state_dict()
                atomic_save(torch, state, output / 'checkpoint_last.pth')
                if improved:
                    atomic_save(torch, state, output / 'checkpoint_best.pth')
                record = {'epoch': epoch, 'seconds': time.time() - started,
                          'train': train_stats, 'val': metrics, 'best_auc': best}
                with (output / 'log.jsonl').open('a') as f:
                    f.write(json.dumps(record) + '\n')
                for key, value in metrics.items():
                    writer.add_scalar('val/' + key, value, epoch)
                writer.flush()
                print(json.dumps(record), flush=True)
            dist.barrier()
    finally:
        if writer is not None:
            writer.close()
        dist.destroy_process_group()


if __name__ == '__main__':
    args = arguments()
    configuration = prepare(args)
    if args.check_only:
        print('Preflight passed (metadata/paths only; GPU, images and weights not loaded).')
    else:
        run(args, configuration)
