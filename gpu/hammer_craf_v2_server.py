#!/usr/bin/env python3
"""CRAF v2: optional decision conditioning and correction penalty on EXISTING caches.

No extraction or HAMMER training. Keep the v1 helper files in this package together.
"""
import argparse
import json
import math
import os
from pathlib import Path
import time

import torch
from torch import nn

import hammer_craf_server as shared
from craf_residual_v2 import FrozenCRAFResidual, correction_penalty

FORMAT = 'frozen_hammer_craf_v2'


def checkpoint_config(args, data, tm, vm):
    return dict(format=FORMAT,
                head=dict(feature_dim=data['text'].shape[1], hidden_dim=args.hidden_dim,
                          dropout=args.dropout, condition_on_decision=args.condition_on_decision),
                learning_rate=args.lr, weight_decay=args.weight_decay,
                correction_lambda=args.correction_lambda, batch_size=args.batch_size,
                seed=args.seed, patience=args.patience, selection_metric=args.selection_metric,
                backbone=shared.backbone_identity(tm['identity']),
                train_manifest_sha256=shared.sha256(Path(args.cache_dir) / 'train.manifest.json'),
                val_manifest_sha256=shared.sha256(Path(args.cache_dir) / 'val.manifest.json'),
                head_source_sha256=shared.sha256(Path(__file__).with_name('craf_residual_v2.py')),
                trainer_source_sha256=shared.sha256(__file__),
                helper_source_sha256=shared.sha256(shared.__file__),
                objective='CE(z_HAMMER + delta) + lambda * mean((delta_fake - delta_real)^2)',
                decision_inputs='p_fake, signed_margin/(1+abs(margin)), entropy/log(2); detached; no labels')


@torch.no_grad()
def validation_summary(model, data, batch_size, device):
    logits = shared.predict(model, data, batch_size, device)
    delta = logits - data['baseline']
    return dict(metrics=shared.metrics(data['labels'], logits),
                margin_correction_mse=float(correction_penalty(delta)),
                mean_abs_margin_correction=float((delta[:, 1] - delta[:, 0]).abs().mean()))


def train(args):
    device = shared.get_device(args.device)
    with shared.output_lock(args.output_dir):
        output = Path(args.output_dir)
        if not args.resume and any((output / n).exists() for n in ['last_craf.pth', 'best_craf.pth', 'history.json']):
            raise ValueError('Output contains an existing run; use a NEW directory or --resume.')
        tr, tm = shared.load_cache(args.cache_dir, 'train')
        val, vm = shared.load_cache(args.cache_dir, 'val')
        if shared.backbone_identity(tm['identity']) != shared.backbone_identity(vm['identity']):
            raise ValueError('Train/validation backbone or preprocessing differs.')
        if tm['identity']['metadata_sha256'] == vm['identity']['metadata_sha256'] or set(tm['news_ids']) & set(vm['news_ids']):
            raise ValueError('Train and validation must use disjoint official splits.')
        if len(torch.unique(val['labels'])) != 2:
            raise ValueError('Validation requires both classes.')
        shared.seed_all(args.seed)
        config = checkpoint_config(args, tr, tm, vm)
        model = FrozenCRAFResidual(**config['head']).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        best, bad, start, history = -float('inf'), 0, 0, []
        if args.resume:
            saved = torch.load(output / 'last_craf.pth', weights_only=True, map_location='cpu')
            if saved['config'] != config:
                raise ValueError('Resume settings, code or caches changed. Use the same v2 run settings.')
            model.load_state_dict(saved['model'], strict=True)
            optimizer.load_state_dict(saved['optimizer'])
            torch.set_rng_state(saved['rng_cpu'])
            if device.type == 'cuda' and saved.get('rng_cuda') is not None:
                torch.cuda.set_rng_state(saved['rng_cuda'], device)
            best, bad, start, history = saved['best'], saved['bad'], saved['epoch'] + 1, saved['history']
        else:
            probe = {k: v[:min(32, len(val['labels']))] for k, v in val.items()}
            torch.testing.assert_close(shared.predict(model, probe, args.batch_size, device),
                                       probe['baseline'], rtol=0, atol=0)
        baseline = shared.metrics(val['labels'], val['baseline'])
        shared.atomic_json(dict(config=config, baseline_validation=baseline, target_epochs=args.epochs), output / 'run.json')
        print('Baseline validation:', json.dumps(baseline), flush=True)
        for epoch in range(start, args.epochs):
            if bad >= args.patience:
                break
            model.train()
            started, n, total_ce, total_penalty, total_loss = time.time(), 0, 0.0, 0.0, 0.0
            for text, image, baseline_logits, labels in shared.loader_for(tr, args.batch_size, True, device):
                optimizer.zero_grad(set_to_none=True)
                corrected, details = model(text.to(device), image.to(device), baseline_logits.to(device), return_details=True)
                ce = nn.functional.cross_entropy(corrected, labels.to(device))
                penalty = correction_penalty(details['residual_logits'])
                loss = ce + args.correction_lambda * penalty
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite training loss')
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                count = len(labels); n += count
                total_ce += ce.item() * count
                total_penalty += penalty.item() * count
                total_loss += loss.item() * count
            summary = validation_summary(model, val, args.batch_size, device)
            score = summary['metrics'][args.selection_metric]
            improved = score > best
            best, bad = max(best, score), 0 if improved else bad + 1
            row = dict(epoch=epoch, train_ce=total_ce / n,
                       train_margin_correction_mse=total_penalty / n,
                       train_penalty_weighted=args.correction_lambda * total_penalty / n,
                       train_loss=total_loss / n, validation=summary['metrics'],
                       validation_correction={k: v for k, v in summary.items() if k != 'metrics'},
                       best=best, patience_count=bad, seconds=time.time() - started)
            history.append(row)
            checkpoint = dict(format=FORMAT, model=model.state_dict(), config=config,
                              epoch=epoch, best=best, bad=bad, optimizer=optimizer.state_dict(), history=history,
                              rng_cpu=torch.get_rng_state(),
                              rng_cuda=torch.cuda.get_rng_state(device) if device.type == 'cuda' else None)
            if improved:
                shared.atomic_torch({k: checkpoint[k] for k in ['format', 'model', 'config', 'epoch', 'best']}, output / 'best_craf.pth')
            shared.atomic_torch(checkpoint, output / 'last_craf.pth')
            shared.atomic_json(history, output / 'history.json')
            print(json.dumps(row), flush=True)
        if not (output / 'best_craf.pth').exists():
            raise RuntimeError('No completed training epoch')
        selected = torch.load(output / 'best_craf.pth', weights_only=True, map_location='cpu')
        model.load_state_dict(selected['model'], strict=True)
        summary = validation_summary(model, val, args.batch_size, device)
        shared.atomic_json(dict(split='val', selected_epoch=selected['epoch'], baseline=baseline,
                                craf=summary['metrics'], correction={k: v for k, v in summary.items() if k != 'metrics'},
                                correction_lambda=args.correction_lambda,
                                condition_on_decision=args.condition_on_decision, test_used=False),
                           output / 'validation_comparison.json')


def evaluate(args):
    output = Path(args.output)
    if output.exists():
        raise ValueError('Evaluation output exists; choose a new file.')
    device = shared.get_device(args.device)
    data, manifest = shared.load_cache(args.cache_dir, args.split)
    checkpoint = torch.load(args.craf_checkpoint, weights_only=True, map_location='cpu')
    if checkpoint.get('format') != FORMAT:
        raise ValueError('This evaluator accepts v2 checkpoints only; use the original script for v1.')
    config = checkpoint['config']
    if shared.backbone_identity(manifest['identity']) != config['backbone']:
        raise ValueError('Evaluation cache backbone/preprocessing differs from training.')
    if shared.sha256(Path(__file__).with_name('craf_residual_v2.py')) != config['head_source_sha256']:
        raise ValueError('v2 architecture source changed since training.')
    for split in ['train', 'val']:
        path = Path(args.cache_dir) / f'{split}.manifest.json'
        if shared.sha256(path) != config[f'{split}_manifest_sha256']:
            raise ValueError(f'{split} manifest changed since training.')
        if args.split == 'test':
            used = json.loads(path.read_text())
            if manifest['identity']['metadata_sha256'] == used['identity']['metadata_sha256'] or set(manifest['news_ids']) & set(used['news_ids']):
                raise ValueError('Test overlaps training/validation metadata or source news IDs.')
    model = FrozenCRAFResidual(**config['head']).to(device)
    model.load_state_dict(checkpoint['model'], strict=True)
    logits = shared.predict(model, data, args.batch_size, device)
    report = dict(split=args.split, count=len(data['labels']), format=FORMAT,
                  craf_checkpoint_sha256=shared.sha256(args.craf_checkpoint), cache_identity=manifest['identity'],
                  selected_epoch=checkpoint['epoch'], head=config['head'], correction_lambda=config['correction_lambda'],
                  baseline=shared.metrics(data['labels'], data['baseline']), craf=shared.metrics(data['labels'], logits),
                  margin_correction_mse=float(correction_penalty(logits - data['baseline'])),
                  note='Frozen HAMMER; binary detection only; no robustness or calibration guarantee.')
    output.parent.mkdir(parents=True, exist_ok=True)
    predictions = output.with_name(output.stem + '_predictions.jsonl')
    temporary = predictions.with_name(predictions.name + '.tmp')
    base_p = data['baseline'].softmax(-1)[:, 1]
    craf_p = logits.softmax(-1)[:, 1]
    with temporary.open('w') as f:
        for i in range(len(data['labels'])):
            f.write(json.dumps(dict(index=int(data['indices'][i]), target=int(data['labels'][i]),
                                   baseline_fake_probability=float(base_p[i]), craf_fake_probability=float(craf_p[i]))) + '\n')
    os.replace(temporary, predictions)
    shared.atomic_json(report, output)
    print(json.dumps(report, indent=2), flush=True)


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    t = sub.add_parser('train')
    t.add_argument('--cache-dir', required=True)
    t.add_argument('--output-dir', required=True)
    t.add_argument('--lr', type=float, default=1e-5)
    t.add_argument('--weight-decay', type=float, default=.01)
    t.add_argument('--epochs', type=int, default=30)
    t.add_argument('--batch-size', type=int, default=512)
    t.add_argument('--hidden-dim', type=int, default=256)
    t.add_argument('--dropout', type=float, default=.2)
    t.add_argument('--seed', type=int, default=42)
    t.add_argument('--patience', type=int, default=8)
    t.add_argument('--selection-metric', choices=['auc', 'macro_f1'], default='auc')
    t.add_argument('--condition-on-decision', action='store_true')
    t.add_argument('--correction-lambda', type=float, default=0.0)
    t.add_argument('--resume', action='store_true')
    t.add_argument('--device', default='cuda')
    e = sub.add_parser('evaluate')
    e.add_argument('--cache-dir', required=True)
    e.add_argument('--craf-checkpoint', required=True)
    e.add_argument('--split', choices=['val', 'test'], default='test')
    e.add_argument('--output', required=True)
    e.add_argument('--batch-size', type=int, default=1024)
    e.add_argument('--device', default='cuda')
    args = p.parse_args()
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        p.error('Use plain python on one GPU, not torchrun.')
    if args.batch_size < 1:
        p.error('--batch-size must be positive')
    if args.command == 'train':
        if (args.epochs < 1 or args.patience < 1 or args.hidden_dim < 1 or
                not math.isfinite(args.lr) or args.lr <= 0 or
                not math.isfinite(args.weight_decay) or args.weight_decay < 0 or
                not math.isfinite(args.correction_lambda) or args.correction_lambda < 0 or
                not 0 <= args.dropout < 1):
            p.error('Invalid training hyperparameters')
    return args


if __name__ == '__main__':
    args_ = arguments()
    {'train': train, 'evaluate': evaluate}[args_.command](args_)
