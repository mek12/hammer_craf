#!/usr/bin/env python3
"""Offline package/source/result integrity checks; no downloads or GPU operations."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess

from summarize_results import summary

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def included():
    """Exclude local files without ever traversing environments, data or .git."""
    import os
    excluded = {'.git', '.idea', '.vscode', '__pycache__', 'venv',
                '.pytest_cache', '.mypy_cache', '.ruff_cache', 'rendered_pages'}
    local_roots = {'official-hammer', 'datasets', 'DGM4', 'checkpoints', 'craf_features', 'gpu_runs'}
    paths = []
    for directory, dirs, files in os.walk(ROOT):
        base = Path(directory)
        dirs[:] = [name for name in dirs if name not in excluded
                   and not name.startswith('.venv') and not name.endswith('.egg-info')
                   and not (base == ROOT and name in local_roots)
                   and not (base / name / 'pyvenv.cfg').exists()]
        for name in files:
            path = base / name
            if (name in {'MANIFEST.sha256', '.DS_Store'} or name.startswith('._')
                    or name == '.env' or (name.startswith('.env.') and name != '.env.example')
                    or path.suffix in {'.pyc', '.pyo', '.tmp', '.lock', '.pem', '.key'}):
                continue
            paths.append(path)
    return sorted(paths)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--write-manifest', action='store_true')
    args = p.parse_args()
    required = ['README.md', '.gitignore', 'requirements-cpu.txt', 'AI_USE.md', 'NOTICE.md',
                'scripts/evidence.py', 'results/user_reported/TRANSCRIPTIONS.json',
                'reports/Kocak_YZM5110_Research.pdf', 'reports/Kocak_YZM5110_Research_TR.pdf',
                'reports/build/report_en.html', 'reports/build/report_tr.html',
                'docs/LITERATURE_MATRIX.md', 'docs/PAPER_READING_GUIDE.md', 'docs/references.bib',
                'docs/GPU_USAGE.md', 'docs/THIRD_PARTY_SLAB_LICENSE.txt',
                'docs/EXTERNAL_SOURCE_VERIFICATION.md', 'docs/REPORT_GENERATION.md',
                'docs/SERVER_RESULTS_ANALYSIS.md', 'docs/SERVER_CODE_AUDIT.md',
                'results/server_analysis.json', 'results/SERVER_EVIDENCE_INDEX.json',
                'results/HAMMER_TRAINING_SUMMARY.md', 'results/hammer_training_analysis.json',
                'results/bootstrap_uncertainty.json', 'results/calibration_analysis.json',
                'results/report_number_check.json', 'scripts/check_report_numbers.py']
    for name in required:
        if not (ROOT / name).is_file():
            raise FileNotFoundError(f'Required artifact missing: {name}')
    sources = json.loads((ROOT / 'gpu/SOURCE_SHA256.json').read_text())
    for item in sources['files']:
        if sha(ROOT / 'gpu' / item['file']) != item['sha256']:
            raise ValueError(f'Preserved source changed: {item["file"]}')
    if (ROOT / 'results/SUMMARY.md').read_text() != summary(ROOT):
        raise ValueError('Result summary differs from source reports')
    evidence_path = ROOT / 'results/SERVER_EVIDENCE_INDEX.json'
    if evidence_path.exists():
        for item in json.loads(evidence_path.read_text())['files']:
            path = ROOT / item['archived']
            if not path.is_file() or sha(path) != item['sha256']:
                raise ValueError(f'Archived server evidence differs: {item["archived"]}')
    analysis = json.loads((ROOT / 'results/server_analysis.json').read_text())
    if analysis['status'] != 'passed' or not all(x['passed'] for x in analysis['checks']):
        raise ValueError('Server result analysis has failed checks')
    hammer_analysis = json.loads((ROOT / 'results/hammer_training_analysis.json').read_text())
    if hammer_analysis['status'] != 'passed' or not all(x['passed'] for x in hammer_analysis['checks']):
        raise ValueError('HAMMER training record analysis has failed checks')
    for path in included():
        if path.suffix in ['.pth', '.pt', '.ckpt', '.safetensors']:
            raise ValueError(f'Unexpected model/cache artifact: {path}')
        if path.suffix == '.py':
            ast.parse(path.read_text(), filename=str(path))
        if path.suffix == '.sh':
            subprocess.run(['bash', '-n', str(path)], check=True)
    manifest = ROOT / 'MANIFEST.sha256'
    if args.write_manifest:
        manifest.write_text(''.join(f'{sha(path)}  {path.relative_to(ROOT).as_posix()}\n' for path in included()))
    elif not manifest.exists():
        raise FileNotFoundError('MANIFEST.sha256 missing; use --write-manifest during packaging.')
    entries = {}
    for line in manifest.read_text().splitlines():
        checksum, relative = line.split('  ', 1)
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT.resolve()) or not path.is_file() or sha(path) != checksum:
            raise ValueError(f'Manifest mismatch: {relative}')
        entries[relative] = checksum
    if set(entries) != {path.relative_to(ROOT).as_posix() for path in included()}:
        raise ValueError('Manifest does not cover exactly the distributed files')
    print(f'PASS: {len(entries)} package files, {len(sources["files"])} preserved sources, 3 consistent result reports.')
    print('Scope: file/source/arithmetic integrity only. No new CUDA experiment or academic submission approval.')


if __name__ == '__main__':
    main()
