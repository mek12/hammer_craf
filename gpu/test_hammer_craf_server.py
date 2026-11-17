"""Portable CPU integration tests using synthetic features, not benchmark results."""
import argparse
import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hammer_craf_server as server


class ServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def caches(self, root):
        torch.manual_seed(7)
        root.mkdir(parents=True, exist_ok=True)
        for offset, split in enumerate(['train', 'val', 'test']):
            n = 12
            identity = dict(format=server.FORMAT, backbone_sha256='fixture-only',
                            split=split, metadata_sha256=split + '-metadata')
            labels = torch.arange(n) % 2
            data = dict(identity=identity, indices=torch.arange(n),
                        text=torch.randn(n, 6), image=torch.randn(n, 6),
                        baseline=torch.zeros(n, 2), labels=labels)
            # Two interleaved shards deliberately exercise full reordering.
            items = []
            for rank in range(2):
                path = root / f'{split}.{rank}.pt'
                shard = {key: value[rank::2] if torch.is_tensor(value) else value for key, value in data.items()}
                server.atomic_torch(shard, path)
                items.append(dict(file=path.name, sha256=server.sha256(path)))
            server.atomic_json(dict(identity=identity, count=n, shards=items,
                                    news_ids=list(range(offset * 100, offset * 100 + n))),
                               root / f'{split}.manifest.json')

    def args(self, cache, output, epochs=3, resume=False):
        return argparse.Namespace(cache_dir=str(cache), output_dir=str(output), epochs=epochs,
                                  batch_size=4, lr=.001, weight_decay=.01, hidden_dim=4,
                                  dropout=.2, seed=42, patience=100, selection_metric='auc',
                                  ablation='full', resume=resume, device='cpu')

    def test_train_resume_evaluate_and_parity(self):
        with tempfile.TemporaryDirectory() as t, contextlib.redirect_stdout(io.StringIO()):
            root = Path(t); cache = root / 'cache'; self.caches(cache)
            data, manifest = server.load_cache(cache, 'train')
            self.assertTrue(torch.equal(data['indices'], torch.arange(12)))
            head = server.FrozenCRAFResidual(feature_dim=6, hidden_dim=4)
            self.assertTrue(torch.equal(server.predict(head, data, 4, torch.device('cpu')), data['baseline']))
            server.train_head(self.args(cache, root / 'continuous', 3))
            server.train_head(self.args(cache, root / 'resumed', 1))
            server.train_head(self.args(cache, root / 'resumed', 3, True))
            a = torch.load(root / 'continuous/last_craf.pth', weights_only=True)
            b = torch.load(root / 'resumed/last_craf.pth', weights_only=True)
            for key in a['model']:
                torch.testing.assert_close(a['model'][key], b['model'][key], rtol=0, atol=0)
            self.assertEqual(b['epoch'], 2)
            self.assertFalse(any('backbone' in key for key in b['model']))
            report = root / 'comparison.json'
            server.evaluate(argparse.Namespace(cache_dir=str(cache), craf_checkpoint=str(root / 'resumed/best_craf.pth'),
                                               split='test', output=str(report), batch_size=4, device='cpu'))
            result = json.loads(report.read_text())
            self.assertEqual(result['count'], 12)
            self.assertIn('baseline', result); self.assertIn('craf', result)
            self.assertEqual(len((root / 'comparison_predictions.jsonl').read_text().splitlines()), 12)
            # Test examples cannot be silently substituted for development examples.
            test_manifest = json.loads((cache / 'test.manifest.json').read_text())
            test_manifest['news_ids'] = [0]
            server.atomic_json(test_manifest, cache / 'test.manifest.json')
            with self.assertRaisesRegex(ValueError, 'shares source news IDs'):
                server.evaluate(argparse.Namespace(cache_dir=str(cache), craf_checkpoint=str(root / 'resumed/best_craf.pth'),
                                                   split='test', output=str(root / 'invalid.json'), batch_size=4, device='cpu'))

    def test_corrupt_and_duplicate_cache_rejected(self):
        with tempfile.TemporaryDirectory() as t, contextlib.redirect_stdout(io.StringIO()):
            cache = Path(t); self.caches(cache)
            path = cache / 'train.0.pt'
            data = torch.load(path, weights_only=True)
            data['indices'][0] = 1
            server.atomic_torch(data, path)
            with self.assertRaisesRegex(ValueError, 'Invalid cache shard'):
                server.load_cache(cache, 'train')
            manifest = json.loads((cache / 'train.manifest.json').read_text())
            manifest['shards'][0]['sha256'] = server.sha256(path)
            server.atomic_json(manifest, cache / 'train.manifest.json')
            with self.assertRaisesRegex(ValueError, 'Missing or duplicated'):
                server.load_cache(cache, 'train')

    def test_eval_hooks_preserve_outputs_and_capture_text_only(self):
        class Bert(nn.Module):
            def forward(self, tensor, mode='text'):
                return SimpleNamespace(last_hidden_state=tensor + (1 if mode == 'text' else 9))

        class Backbone(nn.Module):
            def __init__(self):
                super().__init__(); self.visual_encoder = nn.Identity(); self.text_encoder = nn.Module()
                self.text_encoder.bert = Bert()

            def forward(self, image, text):
                visual = self.visual_encoder(image)
                uni = self.text_encoder.bert(text, mode='text').last_hidden_state
                fused = self.text_encoder.bert(uni, mode='fusion').last_hidden_state
                return visual + fused

        model = Backbone().eval().requires_grad_(False)
        x, y = torch.randn(2, 3, 6), torch.randn(2, 3, 6)
        expected = model(x, y)
        capture = {}; handles = server.feature_hooks(model, capture)
        with torch.inference_mode(): actual = model(x, y)
        for h in handles: h.remove()
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(torch.equal(capture['image'], x[:, 0, :]))
        self.assertTrue(torch.equal(capture['text'], (y + 1)[:, 0, :]))
        self.assertFalse(capture['text'].requires_grad)


if __name__ == '__main__':
    unittest.main()
