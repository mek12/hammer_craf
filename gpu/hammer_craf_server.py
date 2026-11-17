#!/usr/bin/env python3
"""Frozen HAMMER -> cached features -> CRAF residual training and held-out evaluation.

Keep craf_residual.py and train_hammer_4gpu.py beside this file.
Feature extraction supports torchrun; the small cached-feature head trains on ONE GPU.
The official HAMMER repository is read-only. No backbone source patch is needed.
"""
import argparse
from contextlib import contextmanager
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, log_loss, roc_auc_score)

from craf_residual import FrozenCRAFResidual
from train_hammer_4gpu import install_import_compatibility, load_training_config, resolve_repo

FORMAT = 'frozen_hammer_craf_v1'


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(data, path):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def atomic_torch(data, path):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    torch.save(data, temporary)
    os.replace(temporary, path)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(name, local_rank=0):
    device = torch.device(name)
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable; --device cpu is supported for small checks.')
        device = torch.device('cuda', local_rank if device.index is None else device.index)
        torch.cuda.set_device(device)
    return device


@contextmanager
def output_lock(directory):
    import fcntl
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / '.lock').open('w') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f'Another job is writing {directory}') from error
        yield


def metrics(labels, logits):
    labels = labels.cpu().numpy()
    probabilities = logits.float().softmax(-1).cpu().numpy()
    if not np.isfinite(probabilities).all():
        raise ValueError('Nonfinite prediction probabilities')
    predicted = probabilities.argmax(-1)
    return dict(accuracy=float(accuracy_score(labels, predicted)),
                balanced_accuracy=float(balanced_accuracy_score(labels, predicted)),
                macro_f1=float(f1_score(labels, predicted, average='macro', zero_division=0)),
                auc=float(roc_auc_score(labels, probabilities[:, 1])) if len(np.unique(labels)) == 2 else None,
                cross_entropy=float(log_loss(labels, probabilities, labels=[0, 1])),
                confusion_matrix=confusion_matrix(labels, predicted, labels=[0, 1]).tolist())


def feature_hooks(model, capture):
    def image_hook(module, args, output):
        capture['image'] = output[:, 0, :].detach()

    def text_hook(module, args, kwargs, output):
        if kwargs.get('mode') == 'text':
            capture['text'] = output.last_hidden_state[:, 0, :].detach()

    return [model.visual_encoder.register_forward_hook(image_hook),
            model.text_encoder.bert.register_forward_hook(text_hook, with_kwargs=True)]


def build_hammer(args, cfg, tokenizer):
    from models.HAMMER import HAMMER
    from models.xbert import BertForTokenClassification
    from types import SimpleNamespace
    # Provided trained checkpoint supplies all weights; avoid encoder weight downloads.
    previous = BertForTokenClassification.__dict__.get('from_pretrained')
    BertForTokenClassification.from_pretrained = classmethod(
        lambda cls, *a, **kw: cls(config=kw['config'], label_smoothing=kw.get('label_smoothing', 0.0)))
    try:
        model = HAMMER(args=SimpleNamespace(token_momentum=True), config=cfg,
                       text_encoder=args.text_encoder, tokenizer=tokenizer, init_deit=False)
    finally:
        if previous is None:
            del BertForTokenClassification.from_pretrained
        else:
            BertForTokenClassification.from_pretrained = previous
    # Omit mmap for compatibility with the older PyTorch environments in use.
    checkpoint = torch.load(args.hammer_checkpoint, weights_only=True, map_location='cpu')
    if not isinstance(checkpoint.get('epoch'), int) or isinstance(checkpoint['epoch'], bool) or checkpoint['epoch'] < 0:
        raise ValueError('A completed HAMMER training checkpoint with a nonnegative epoch is required.')
    if isinstance(checkpoint.get('config'), dict):
        for key in ['image_res', 'vision_width', 'embed_dim', 'max_words']:
            if key in checkpoint['config'] and checkpoint['config'][key] != cfg[key]:
                raise ValueError(f'Inference config differs from HAMMER training: {key}')
    # ALBEF is deliberately NOT accepted here: all HAMMER task heads must be trained.
    model.load_state_dict(checkpoint['model'], strict=True)
    return model.eval().requires_grad_(False)


def extract(args):
    import torch.distributed as dist
    from torchvision import transforms
    from PIL import Image
    from transformers import BertTokenizerFast
    rank, world, local = (int(os.environ.get(key, default)) for key, default in
                          [('RANK', '0'), ('WORLD_SIZE', '1'), ('LOCAL_RANK', '0')])
    device = get_device(args.device, local)
    if world > 1 and (device.type != 'cuda' or torch.device(args.device).index is not None):
        raise ValueError('Distributed extraction requires --device cuda (without a device index).')
    if world > 1:
        dist.init_process_group('nccl', init_method='env://', timeout=timedelta(hours=4))
    try:
        repo = resolve_repo(args.repo_dir)
        sys.path.insert(0, str(repo))
        install_import_compatibility()
        import train as upstream
        from dataset.dataset import DGM4_Dataset
        cfg_path = Path(args.config)
        cfg = load_training_config(cfg_path if cfg_path.is_absolute() else repo / cfg_path)
        cfg.pop('dataset_division', None)  # Full official split, never a configured subset.
        bert = Path(cfg['bert_config'])
        cfg['bert_config'] = str(bert if bert.is_absolute() else repo / bert)
        data_root = Path(args.data_root).expanduser().resolve()
        cache = Path(args.cache_dir).expanduser().resolve()
        cache.mkdir(parents=True, exist_ok=True)
        seed_all(42)
        tokenizer = BertTokenizerFast.from_pretrained(args.text_encoder)
        identity = dict(format=FORMAT, backbone_sha256=sha256(args.hammer_checkpoint),
                        bert_config_sha256=sha256(cfg['bert_config']),
                        tokenizer_sha256=hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest(),
                        image_res=cfg['image_res'], max_words=cfg['max_words'],
                        extraction='official_eval_transform_pre_caption_remove_SEP_prefusion_CLS_v1',
                        source_sha256={name: sha256(repo / name) for name in
                                       ['models/HAMMER.py', 'models/xbert.py', 'models/vit.py',
                                        'dataset/dataset.py', 'dataset/utils.py', 'train.py']},
                        extractor_sha256=sha256(__file__))
        transform = transforms.Compose([
            transforms.Resize((cfg['image_res'], cfg['image_res']), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((.48145466, .4578275, .40821073), (.26862954, .26130258, .27577711))])
        print(f'Rank {rank}/{world}: loading frozen HAMMER on {device}', flush=True)
        model = build_hammer(args, cfg, tokenizer).to(device)
        capture = {}
        handles = feature_hooks(model, capture)
        try:
            for split in args.splits:
                metadata = data_root / 'DGM4/metadata' / f'{split}.json'
                split_identity = dict(identity, split=split, metadata_sha256=sha256(metadata))
                dataset = DGM4_Dataset(config=cfg, ann_file=[str(metadata)], transform=transform,
                                       max_words=cfg['max_words'], is_train=False)
                dataset.root_dir = str(data_root)
                indices = list(range(rank, len(dataset), world))  # No padding or dropped examples.
                if not indices:
                    raise ValueError('More extraction processes than examples.')
                shard = cache / f'{split}.w{world}.r{rank}.pt'
                with output_lock(cache / f'.{split}.w{world}.r{rank}'):
                    if shard.exists():
                        saved = torch.load(shard, weights_only=True, map_location='cpu')
                        if saved['identity'] != split_identity or saved['indices'].tolist() != indices:
                            raise ValueError(f'Existing cache has different inputs: {shard}; use a fresh cache directory.')
                        print(f'Rank {rank}: reusing {shard}', flush=True)
                        del saved
                    else:
                        loader = DataLoader(Subset(dataset, indices), batch_size=args.batch_size,
                                            shuffle=False, drop_last=False, num_workers=args.workers,
                                            pin_memory=device.type == 'cuda')
                        collected = {key: [] for key in ['text', 'image', 'baseline', 'labels']}
                        processed, started = 0, time.time()
                        with torch.inference_mode():
                            for image, kinds, captions, boxes, words, width, height in loader:
                                encoded = tokenizer(captions, max_length=128, truncation=True,
                                                    add_special_tokens=True, return_attention_mask=True,
                                                    return_token_type_ids=False)
                                text, _ = upstream.text_input_adjust(encoded, words, device)
                                capture.clear()
                                outputs = model(image.to(device, non_blocking=True), kinds, text,
                                                None, None, is_train=False)
                                if set(capture) != {'text', 'image'}:
                                    raise RuntimeError('Could not capture pre-fusion HAMMER embeddings.')
                                values = dict(text=capture['text'].float().cpu(),
                                              image=capture['image'].float().cpu(),
                                              baseline=outputs[0].float().cpu(),
                                              labels=torch.tensor([int(k != 'orig') for k in kinds]))
                                for key, value in values.items():
                                    collected[key].append(value)
                                processed += len(kinds)
                                if processed % (args.batch_size * 50) == 0:
                                    print(f'Rank {rank} {split}: {processed}/{len(indices)} '
                                          f'({processed / (time.time() - started):.1f} examples/s)', flush=True)
                        tensors = {key: torch.cat(value) for key, value in collected.items()}
                        if not all(torch.isfinite(v).all() for v in tensors.values()):
                            raise ValueError('Nonfinite extracted values')
                        atomic_torch(dict(identity=split_identity, indices=torch.tensor(indices), **tensors), shard)
                if world > 1:
                    dist.barrier()
                if rank == 0:
                    shards = [cache / f'{split}.w{world}.r{r}.pt' for r in range(world)]
                    manifest = dict(identity=split_identity, count=len(dataset),
                                    news_ids=sorted({int(row['id']) for row in dataset.ann}),
                                    shards=[dict(file=p.name, sha256=sha256(p)) for p in shards])
                    atomic_json(manifest, cache / f'{split}.manifest.json')
                    print(f'{split} cache complete: {len(dataset)} records', flush=True)
                if world > 1:
                    dist.barrier()
        finally:
            for handle in handles:
                handle.remove()
    finally:
        if world > 1:
            dist.destroy_process_group()


def load_cache(directory, split):
    directory = Path(directory)
    manifest_path = directory / f'{split}.manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest['identity']['format'] != FORMAT or manifest['identity']['split'] != split:
        raise ValueError(f'Invalid manifest: {manifest_path}')
    parts = []
    for item in manifest['shards']:
        path = directory / item['file']
        if path.parent.resolve() != directory.resolve() or sha256(path) != item['sha256']:
            raise ValueError(f'Invalid cache shard: {path}')
        data = torch.load(path, weights_only=True, map_location='cpu')
        if data['identity'] != manifest['identity']:
            raise ValueError(f'Shard identity mismatch: {path}')
        parts.append(data)
    data = {key: torch.cat([p[key] for p in parts]) for key in ['indices', 'text', 'image', 'baseline', 'labels']}
    order = data['indices'].argsort()
    data = {key: value[order] for key, value in data.items()}
    n = manifest['count']
    if not torch.equal(data['indices'], torch.arange(n)):
        raise ValueError('Missing or duplicated cache record indices')
    dimension = data['text'].shape[1]
    if data['text'].shape != (n, dimension) or data['image'].shape != (n, dimension):
        raise ValueError('Invalid embedding shapes')
    if data['baseline'].shape != (n, 2) or data['labels'].shape != (n,):
        raise ValueError('Invalid logits/label shapes')
    if not all(torch.isfinite(v).all() for v in data.values()):
        raise ValueError('Nonfinite cached values')
    if not ((data['labels'] == 0) | (data['labels'] == 1)).all():
        raise ValueError('Expected orig=0, manipulated=1 labels')
    return data, manifest


def backbone_identity(identity):
    return {k: v for k, v in identity.items() if k not in ['split', 'metadata_sha256']}


def loader_for(data, batch_size, shuffle, device):
    return DataLoader(TensorDataset(data['text'], data['image'], data['baseline'], data['labels']),
                      batch_size=batch_size, shuffle=shuffle, num_workers=0,
                      pin_memory=device.type == 'cuda', drop_last=False)


@torch.no_grad()
def predict(model, data, batch_size, device):
    model.eval()
    logits = []
    for text, image, baseline, labels in loader_for(data, batch_size, False, device):
        logits.append(model(text.to(device), image.to(device), baseline.to(device)).cpu())
    return torch.cat(logits)


def head_config(args, feature_dim):
    return dict(feature_dim=feature_dim, hidden_dim=args.hidden_dim, dropout=args.dropout,
                use_reliability=args.ablation != 'no_reliability',
                use_consistency=args.ablation != 'no_consistency',
                use_interactions=args.ablation != 'no_interactions')


def train_head(args):
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise ValueError('Train the cached-feature head with plain python on one GPU, not torchrun.')
    device = get_device(args.device)
    with output_lock(args.output_dir):
        output = Path(args.output_dir)
        if not args.resume and any((output / name).exists() for name in ['last_craf.pth', 'best_craf.pth', 'history.json']):
            raise ValueError('Output already contains training results. Use --resume or a fresh output directory.')
        train, tm = load_cache(args.cache_dir, 'train')
        val, vm = load_cache(args.cache_dir, 'val')
        if backbone_identity(tm['identity']) != backbone_identity(vm['identity']):
            raise ValueError('Train and val were extracted from different backbone/preprocessing inputs.')
        if tm['identity']['metadata_sha256'] == vm['identity']['metadata_sha256']:
            raise ValueError('Train and val metadata must be different.')
        if set(tm['news_ids']) & set(vm['news_ids']):
            raise ValueError('Train and validation share source news IDs; use the official disjoint splits.')
        if len(torch.unique(val['labels'])) != 2:
            raise ValueError('Validation must include both binary classes.')
        seed_all(args.seed)
        config = dict(format=FORMAT, head=head_config(args, train['text'].shape[1]),
                      learning_rate=args.lr, weight_decay=args.weight_decay,
                      batch_size=args.batch_size, seed=args.seed, selection_metric=args.selection_metric,
                      patience=args.patience, backbone=backbone_identity(tm['identity']),
                      train_manifest_sha256=sha256(Path(args.cache_dir) / 'train.manifest.json'),
                      val_manifest_sha256=sha256(Path(args.cache_dir) / 'val.manifest.json'),
                      head_source_sha256=sha256(Path(__file__).with_name('craf_residual.py')),
                      trainer_source_sha256=sha256(__file__),
                      objective='CE(HAMMER_binary_logits + CRAF_residual); binary labels only')
        model = FrozenCRAFResidual(**config['head']).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        best, bad, start, history = -float('inf'), 0, 0, []
        if args.resume:
            saved = torch.load(output / 'last_craf.pth', weights_only=True, map_location='cpu')
            if saved['config'] != config:
                raise ValueError('Resume configuration or cache identity changed.')
            model.load_state_dict(saved['model'], strict=True)
            optimizer.load_state_dict(saved['optimizer'])
            torch.set_rng_state(saved['rng_cpu'])
            if device.type == 'cuda' and saved.get('rng_cuda') is not None:
                torch.cuda.set_rng_state(saved['rng_cuda'], device)
            best, bad, start = saved['best'], saved['bad'], saved['epoch'] + 1
            history = saved['history']
        else:
            # This invariant guarantees the new head does not initially disturb HAMMER.
            probe = {k: v[:min(32, len(val['labels']))] for k, v in val.items()}
            torch.testing.assert_close(predict(model, probe, args.batch_size, device), probe['baseline'], rtol=0, atol=0)
        baseline = metrics(val['labels'], val['baseline'])
        atomic_json(dict(config=config, baseline_validation=baseline, target_epochs=args.epochs), output / 'run.json')
        print('Frozen HAMMER validation:', json.dumps(baseline), flush=True)
        for epoch in range(start, args.epochs):
            if bad >= args.patience:
                break
            started, loss_sum, count = time.time(), 0.0, 0
            model.train()
            for text, image, logits, labels in loader_for(train, args.batch_size, True, device):
                optimizer.zero_grad(set_to_none=True)
                corrected = model(text.to(device), image.to(device), logits.to(device))
                loss = nn.functional.cross_entropy(corrected, labels.to(device))
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite CRAF training loss')
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                loss_sum += loss.item() * len(labels)
                count += len(labels)
            validation = metrics(val['labels'], predict(model, val, args.batch_size, device))
            score = validation[args.selection_metric]
            improved = score > best
            best, bad = max(best, score), 0 if improved else bad + 1
            row = dict(epoch=epoch, train_loss=loss_sum / count, validation=validation,
                       seconds=time.time() - started, best=best, patience_count=bad)
            history.append(row)
            checkpoint = dict(format=FORMAT, model=model.state_dict(), config=config,
                              epoch=epoch, best=best, bad=bad, optimizer=optimizer.state_dict(),
                              rng_cpu=torch.get_rng_state(),
                              rng_cuda=torch.cuda.get_rng_state(device) if device.type == 'cuda' else None,
                              history=history)
            if improved:
                atomic_torch({k: checkpoint[k] for k in ['format', 'model', 'config', 'epoch', 'best']},
                             output / 'best_craf.pth')
            # Commit last only once an improved best has been saved successfully.
            atomic_torch(checkpoint, output / 'last_craf.pth')
            atomic_json(history, output / 'history.json')
            print(json.dumps(row), flush=True)
        if not (output / 'best_craf.pth').is_file():
            raise RuntimeError('No completed training epoch/best checkpoint.')
        best_state = torch.load(output / 'best_craf.pth', weights_only=True, map_location='cpu')
        model.load_state_dict(best_state['model'], strict=True)
        corrected = metrics(val['labels'], predict(model, val, args.batch_size, device))
        atomic_json(dict(split='val', selected_epoch=best_state['epoch'], baseline=baseline,
                         craf=corrected, test_used=False), output / 'validation_comparison.json')


def evaluate(args):
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise ValueError('Evaluate the small head with plain python, not torchrun.')
    output = Path(args.output)
    if output.exists():
        raise ValueError('Evaluation output already exists; choose a new file.')
    device = get_device(args.device)
    data, manifest = load_cache(args.cache_dir, args.split)
    checkpoint = torch.load(args.craf_checkpoint, weights_only=True, map_location='cpu')
    config = checkpoint['config']
    if checkpoint['format'] != FORMAT or backbone_identity(manifest['identity']) != config['backbone']:
        raise ValueError('Evaluation cache does not match the CRAF backbone/preprocessing.')
    if sha256(Path(__file__).with_name('craf_residual.py')) != config['head_source_sha256']:
        raise ValueError('CRAF architecture source changed since training.')
    if args.split == 'test':
        train_path = Path(args.cache_dir) / 'train.manifest.json'
        val_path = Path(args.cache_dir) / 'val.manifest.json'
        if sha256(train_path) != config['train_manifest_sha256'] or sha256(val_path) != config['val_manifest_sha256']:
            raise ValueError('Train/validation manifests changed since CRAF training.')
        train_manifest = json.loads(train_path.read_text())
        val_manifest = json.loads(val_path.read_text())
        if manifest['identity']['metadata_sha256'] in [train_manifest['identity']['metadata_sha256'], val_manifest['identity']['metadata_sha256']]:
            raise ValueError('Test metadata must differ from training/development metadata.')
        if set(manifest['news_ids']) & (set(train_manifest['news_ids']) | set(val_manifest['news_ids'])):
            raise ValueError('Test shares source news IDs with training/development splits.')
    model = FrozenCRAFResidual(**config['head']).to(device)
    model.load_state_dict(checkpoint['model'], strict=True)
    logits = predict(model, data, args.batch_size, device)
    report = dict(split=args.split, count=len(data['labels']),
                  craf_checkpoint_sha256=sha256(args.craf_checkpoint), cache_identity=manifest['identity'],
                  selected_epoch=checkpoint['epoch'], baseline=metrics(data['labels'], data['baseline']),
                  craf=metrics(data['labels'], logits),
                  note='Binary detection only. Other HAMMER heads unchanged; no robustness claim.')
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(report, output)
    probabilities = logits.softmax(-1)[:, 1]
    baseline_probabilities = data['baseline'].softmax(-1)[:, 1]
    prediction_path = output.with_name(output.stem + '_predictions.jsonl')
    with prediction_path.open('w') as f:
        for i in range(len(data['labels'])):
            f.write(json.dumps(dict(index=int(data['indices'][i]), target=int(data['labels'][i]),
                                   baseline_fake_probability=float(baseline_probabilities[i]),
                                   craf_fake_probability=float(probabilities[i]))) + '\n')
    print(json.dumps(report, indent=2), flush=True)


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    e = sub.add_parser('extract', help='Run frozen HAMMER once to cache features; torchrun optional')
    e.add_argument('--repo-dir', required=True)
    e.add_argument('--config', default='configs/train.yaml')
    e.add_argument('--hammer-checkpoint', required=True)
    e.add_argument('--data-root', required=True)
    e.add_argument('--cache-dir', required=True)
    e.add_argument('--splits', nargs='+', choices=['train', 'val', 'test'], default=['train', 'val'])
    e.add_argument('--text-encoder', default='bert-base-uncased')
    e.add_argument('--batch-size', type=int, default=32)
    e.add_argument('--workers', type=int, default=4)
    e.add_argument('--device', default='cuda')
    t = sub.add_parser('train', help='Train CRAF only; no HAMMER weights/forward required')
    t.add_argument('--cache-dir', required=True)
    t.add_argument('--output-dir', required=True)
    t.add_argument('--epochs', type=int, default=30)
    t.add_argument('--batch-size', type=int, default=512)
    t.add_argument('--lr', type=float, default=1e-4)
    t.add_argument('--weight-decay', type=float, default=.01)
    t.add_argument('--hidden-dim', type=int, default=256)
    t.add_argument('--dropout', type=float, default=.2)
    t.add_argument('--seed', type=int, default=42)
    t.add_argument('--patience', type=int, default=5)
    t.add_argument('--selection-metric', choices=['auc', 'macro_f1'], default='auc')
    t.add_argument('--ablation', choices=['full', 'no_reliability', 'no_consistency', 'no_interactions'], default='full')
    t.add_argument('--resume', action='store_true')
    t.add_argument('--device', default='cuda')
    v = sub.add_parser('evaluate', help='Explicit baseline/CRAF comparison, normally on held-out test')
    v.add_argument('--cache-dir', required=True)
    v.add_argument('--craf-checkpoint', required=True)
    v.add_argument('--split', choices=['val', 'test'], default='test')
    v.add_argument('--output', required=True)
    v.add_argument('--batch-size', type=int, default=1024)
    v.add_argument('--device', default='cuda')
    args = p.parse_args()
    if args.batch_size < 1:
        p.error('--batch-size must be positive')
    if args.command == 'train':
        if args.epochs < 1 or args.patience < 1 or args.lr <= 0 or args.hidden_dim < 1 or not 0 <= args.dropout < 1:
            p.error('Invalid training hyperparameters')
    if args.command == 'extract' and args.workers < 0:
        p.error('--workers must be nonnegative')
    return args


if __name__ == '__main__':
    arguments_ = arguments()
    {'extract': extract, 'train': train_head, 'evaluate': evaluate}[arguments_.command](arguments_)
