"""Tiny synthetic CPU integration tests for CRAF v2; not benchmark results."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import hammer_craf_v2_server as v2
import test_hammer_craf_server as fixtures


class V2ServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def args(self, cache, output, epochs, resume=False, condition=True, penalty=.1):
        return argparse.Namespace(cache_dir=str(cache), output_dir=str(output), epochs=epochs,
                                  batch_size=4, lr=.001, weight_decay=.01, hidden_dim=4,
                                  dropout=.2, seed=42, patience=100, selection_metric='auc',
                                  resume=resume, device='cpu', condition_on_decision=condition,
                                  correction_lambda=penalty)

    def test_train_resume_evaluate_reuses_v1_cache(self):
        with tempfile.TemporaryDirectory() as t, contextlib.redirect_stdout(io.StringIO()):
            root = Path(t); cache = root / 'cache'; fixtures.ServerTests().caches(cache)
            v2.train(self.args(cache, root / 'continuous', 3))
            v2.train(self.args(cache, root / 'resume', 1))
            v2.train(self.args(cache, root / 'resume', 3, resume=True))
            a = torch.load(root / 'continuous/last_craf.pth', weights_only=True)
            b = torch.load(root / 'resume/last_craf.pth', weights_only=True)
            for key in a['model']:
                torch.testing.assert_close(a['model'][key], b['model'][key], rtol=0, atol=0)
            for row in a['history']:
                self.assertAlmostEqual(row['train_loss'], row['train_ce'] + .1 * row['train_margin_correction_mse'], places=6)
            self.assertTrue(a['config']['head']['condition_on_decision'])
            report = root / 'test.json'
            v2.evaluate(argparse.Namespace(cache_dir=str(cache), craf_checkpoint=str(root / 'resume/best_craf.pth'),
                                           split='test', output=str(report), batch_size=4, device='cpu'))
            result = json.loads(report.read_text())
            self.assertEqual(result['count'], 12); self.assertEqual(result['correction_lambda'], .1)
            self.assertIn('margin_correction_mse', result)
            self.assertEqual(len((root / 'test_predictions.jsonl').read_text().splitlines()), 12)
            with self.assertRaisesRegex(ValueError, 'settings, code or caches changed'):
                v2.train(self.args(cache, root / 'resume', 4, resume=True, penalty=.2))

    def test_control_and_penalty_can_run_without_conditioning(self):
        with tempfile.TemporaryDirectory() as t, contextlib.redirect_stdout(io.StringIO()):
            root = Path(t); cache = root / 'cache'; fixtures.ServerTests().caches(cache)
            for penalty in [0., .1]:
                output = root / str(penalty)
                v2.train(self.args(cache, output, 1, condition=False, penalty=penalty))
                report = json.loads((output / 'validation_comparison.json').read_text())
                self.assertFalse(report['test_used'])
                self.assertFalse(report['condition_on_decision'])


if __name__ == '__main__':
    unittest.main()
