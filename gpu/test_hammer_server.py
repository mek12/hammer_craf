#!/usr/bin/env python3
"""Evaluate a trained HAMMER checkpoint on the complete official DGM4 test split.

Run with plain python on one GPU. Keep train_hammer_4gpu.py beside this file.
Uses the same upstream evaluation function as the training launcher; no training.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

from train_hammer_4gpu import (METRICS, install_import_compatibility,
                                load_training_config, resolve_repo)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo-dir')
    p.add_argument('--config', default='configs/train.yaml')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--data-root', required=True, help='Parent directory containing DGM4/')
    p.add_argument('--output', required=True, help='New JSON output path')
    p.add_argument('--text-encoder', default='bert-base-uncased')
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--device', default='cuda')
    p.add_argument('--check-only', action='store_true')
    a = p.parse_args()
    if a.batch_size < 1 or a.workers < 0:
        p.error('batch-size must be positive; workers must be nonnegative')
    return a


def run(a):
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise ValueError('Run with plain python on one GPU, not torchrun.')
    root = resolve_repo(a.repo_dir)
    sys.path.insert(0, str(root))
    config_path = Path(a.config)
    cfg = load_training_config(config_path if config_path.is_absolute() else root / config_path)
    cfg.pop('dataset_division', None)
    bert = Path(cfg['bert_config'])
    cfg['bert_config'] = str(bert if bert.is_absolute() else root / bert)
    data_root = Path(a.data_root).expanduser().resolve()
    metadata = data_root / 'DGM4/metadata/test.json'
    checkpoint_path = Path(a.checkpoint).expanduser().resolve()
    output = Path(a.output).expanduser().resolve()
    if output.exists():
        raise ValueError(f'Output exists; choose a new file: {output}')
    for path in [metadata, checkpoint_path, Path(cfg['bert_config'])]:
        if not path.is_file():
            raise FileNotFoundError(path)
    with metadata.open() as f:
        annotations = json.load(f)
    if not isinstance(annotations, list) or not annotations:
        raise ValueError('Expected a nonempty test annotation list')
    if len({int(row['fake_cls'] != 'orig') for row in annotations}) != 2:
        raise ValueError('The test split must include both real and manipulated examples.')
    print(f'Test metadata: {metadata}\nSamples: {len(annotations)}\nCheckpoint: {checkpoint_path}', flush=True)
    if a.check_only:
        print('Path/metadata checks passed. Weights, images and GPU not checked.')
        return
    del annotations
    import torch
    from PIL import Image
    from torchvision import transforms
    from torch.utils.data import DataLoader
    from transformers import BertTokenizerFast
    install_import_compatibility()
    import train as upstream
    from dataset.dataset import DGM4_Dataset
    from models.HAMMER import HAMMER
    from models.xbert import BertForTokenClassification
    device = torch.device(a.device)
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable. --device cpu is supported for small compatibility checks.')
        torch.cuda.set_device(device.index or 0)
    a.log, a.distributed, a.token_momentum = True, False, True
    tokenizer = BertTokenizerFast.from_pretrained(a.text_encoder)
    previous = BertForTokenClassification.__dict__.get('from_pretrained')
    BertForTokenClassification.from_pretrained = classmethod(
        lambda cls, *args, **kw: cls(config=kw['config'], label_smoothing=kw.get('label_smoothing', 0.0)))
    try:
        model = HAMMER(args=a, config=cfg, text_encoder=a.text_encoder,
                       tokenizer=tokenizer, init_deit=False)
    finally:
        if previous is None:
            del BertForTokenClassification.from_pretrained
        else:
            BertForTokenClassification.from_pretrained = previous
    checkpoint_sha = digest(checkpoint_path)
    # Older PyTorch versions forward mmap to pickle instead of recognizing it.
    payload = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    epoch = payload.get('epoch')
    if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0:
        raise ValueError('Provide a trained HAMMER checkpoint with a completed epoch, not ALBEF.')
    saved_cfg = payload.get('config', {})
    for key in ['image_res', 'vision_width', 'embed_dim', 'max_words']:
        if key in saved_cfg and cfg[key] != saved_cfg[key]:
            raise ValueError(f'Test config differs from training: {key}')
    model.load_state_dict(payload['model'], strict=True)
    if digest(checkpoint_path) != checkpoint_sha:
        raise RuntimeError('Checkpoint changed while loading. Test an immutable snapshot or wait for training to finish.')
    del payload
    model.eval().requires_grad_(False).to(device)
    transform = transforms.Compose([
        transforms.Resize((cfg['image_res'], cfg['image_res']), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((.48145466, .4578275, .40821073), (.26862954, .26130258, .27577711))])
    dataset = DGM4_Dataset(config=cfg, ann_file=[str(metadata)], transform=transform,
                           max_words=cfg['max_words'], is_train=False)
    dataset.root_dir = str(data_root)
    loader = DataLoader(dataset, batch_size=a.batch_size, shuffle=False, drop_last=False,
                        num_workers=a.workers, pin_memory=device.type == 'cuda')
    started = time.time()
    values = upstream.evaluation(a, model, loader, tokenizer, device, cfg)
    if len(values) != len(METRICS):
        raise RuntimeError('Unexpected upstream metric output')
    results = {key: float(value) for key, value in zip(METRICS, values)}
    if not all(math.isfinite(v) for v in results.values()):
        raise RuntimeError('Nonfinite evaluation metrics; inspect data and predictions.')
    report = dict(split='test', sample_count=len(dataset), checkpoint=str(checkpoint_path),
                  checkpoint_sha256=checkpoint_sha, checkpoint_epoch_zero_based=epoch,
                  metadata=str(metadata), metadata_sha256=digest(metadata), strict_load=True,
                  device=str(device), batch_size=a.batch_size, seconds=time.time() - started,
                  metrics=results,
                  source_sha256={name: digest(root / name) for name in
                                 ['train.py', 'models/HAMMER.py', 'dataset/dataset.py']},
                  note='Full supplied test.json, official train.evaluation, no training or checkpoint selection.')
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + '.tmp')
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, output)
    print(json.dumps(results, indent=2), flush=True)
    print(f'Saved test report: {output}', flush=True)


if __name__ == '__main__':
    run(arguments())
