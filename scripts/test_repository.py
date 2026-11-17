"""Regression checks for manifest scope and historical-result fingerprints."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from evidence import normalized_sha
from summarize_results import load_results
import verify_package


class RepositoryTests(unittest.TestCase):
    def test_manifest_ignores_local_state_but_keeps_nested_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            keep = ['README.md', '.gitignore', 'gpu/model.py',
                    'results/server_records/run/craf_features/train.manifest.json']
            ignore = ['.git/config', '.venv/lib/package.py', '.env',
                      'gpu/__pycache__/model.pyc', 'gpu_runs/run/history.json',
                      'qa/rendered_pages/page.png', 'datasets/train.json',
                      'official-hammer/train.py', '.DS_Store']
            for name in keep + ignore:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('fixture')
            with patch.object(verify_package, 'ROOT', root):
                actual = {str(p.relative_to(root)) for p in verify_package.included()}
            self.assertEqual(actual, set(keep))

    def test_fingerprint_accepts_json_formatting_but_detects_value_changes(self):
        first = json.loads('{"metric": 0.9, "count": 10}')
        reformatted = json.loads('{\n  "count": 10, "metric": 0.9\n}')
        self.assertEqual(normalized_sha(first), normalized_sha(reformatted))
        self.assertNotEqual(normalized_sha(first), normalized_sha(dict(first, metric=0.8)))

    def test_summary_rejects_changed_historical_fingerprint(self):
        with patch('summarize_results.transcription_record', return_value={
                'normalized_json_sha256': '0' * 64}):
            with self.assertRaisesRegex(ValueError, 'Historical transcription fingerprint differs'):
                load_results()


if __name__ == '__main__':
    unittest.main()
