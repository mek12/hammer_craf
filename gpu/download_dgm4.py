#!/usr/bin/env python3
"""Download, unpack and validate the COMPLETE official DGM4 dataset.

Python 3.9+ and curl only; no PyTorch/Hugging Face package required.
Run once with plain python, NOT torchrun. Interrupted downloads can be resumed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import zipfile

REVISION = '9efd3f1ca14cb1dac1692cdce0beea6844b968a6'
BASE = f'https://huggingface.co/datasets/rshaojimmy/DGM4/resolve/{REVISION}'
# Sizes and LFS SHA-256 hashes verified against the official HF API.
FILES = [
    ('metadata/train.json', 144878709, 'dafd4fe101d3de63b433759fae79b14b465294f7be73f26c9f3bb13b8de87704'),
    ('metadata/val.json', 15985338, 'd4377ecc19cb6528e36c77401482cbfffaa57977faee2c6124e3b1bdda045520'),
    ('metadata/test.json', 36769032, 'b58b29c4cf0f40cdf791189181ad57821f9359e358e93683cc702db3a41c0a18'),
    ('origin/bbc.zip', 1374451578, 'f0bcb036bb11016802b0d7eea202820d61376372045d1e24b86d9b8d0c19d449'),
    ('origin/guardian.zip', 1893273813, '78b7c05830742dc8ae23c0bbc32d4383bcbcb19647bd513fd854a6826576c4fd'),
    ('origin/usa_today.zip', 572095335, '0cc27827eeaac0483eaab12b4bb58989dc6ba360ce07be6b83c59717bdca0991'),
    ('origin/washington_post.zip', 405746497, 'c7b7cf1dd358c51d1ee76fd4c5e9d691b16ca20a498b9c08718a9378186befcf'),
    ('manipulation/HFGI.zip', 1432870701, '6424944272ae58193fd64404d3e1bcf3dd0cd268f01d4a4af5b6d29b198c5ced'),
    ('manipulation/StyleCLIP.zip', 1429202710, '2a712d3889823621766f0200e0ddc343b11b202d08f52b0589354de8f00a87dc'),
    ('manipulation/infoswap.zip', 2035798675, 'e2119a760551c44f5a9da2e601c5d902bde75eb97e91f073df2a5a7107e57769'),
    ('manipulation/simswap.zip', 1385353213, '7fbc0e606eb6eec0b7541ce71dc92d020e4677adeb3f71b5ec89e335a33a2465'),
]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def verify_download(path, size, sha):
    if path.stat().st_size != size or digest(path) != sha:
        raise RuntimeError(f'Invalid size/SHA-256: {path}. Move this file aside and rerun.')


def download(relative, size, sha, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        print(f'Verifying existing: {relative}', flush=True)
        verify_download(path, size, sha)
        return
    partial = path.with_name(path.name + '.part')
    current = partial.stat().st_size if partial.exists() else 0
    if current > size:
        raise RuntimeError(f'Oversized partial download: {partial}; move aside and rerun.')
    if current < size:
        if shutil.disk_usage(path.parent).free < size - current + 256 * 1024**2:
            raise RuntimeError(f'Insufficient free disk space for {relative}')
        print(f'Downloading {relative} ({size / 1e9:.2f} GB), resume at {current} bytes', flush=True)
        # Argument list, no shell interpolation. curl keeps partial data on errors.
        subprocess.run(['curl', '--fail', '--location', '--retry', '8', '--retry-delay', '5',
                        '--connect-timeout', '30', '--speed-limit', '1024', '--speed-time', '120',
                        '--continue-at', '-', '--output', str(partial), f'{BASE}/{relative}'], check=True)
    print(f'Checking SHA-256: {relative}', flush=True)
    verify_download(partial, size, sha)
    os.replace(partial, path)


def destination_for(member, archive_relative, root):
    """Normalize known archive prefixes without changing metadata paths."""
    parts = PurePosixPath(member).parts
    if not parts or member.startswith('/') or '\\' in member or '..' in parts:
        raise ValueError(f'Unsafe archive member: {member}')
    if '__MACOSX' in parts or parts[-1].startswith('._') or parts[-1] == '.DS_Store':
        return None
    group, archive_name = archive_relative.split('/')
    folder = Path(archive_name).stem
    if folder in parts:
        remaining = parts[parts.index(folder) + 1:]
    else:
        # A flat archive or an archive containing just numeric subdirectories.
        remaining = parts
    if not remaining:
        raise ValueError(f'No file path after archive prefix: {member}')
    destination = root / 'DGM4' / group / folder / Path(*remaining)
    if not destination.resolve().is_relative_to(root.resolve()):
        raise ValueError(f'Archive target escapes data root: {member}')
    return destination


def unpack(path, relative, root, marker, sha, reextract=False):
    if not reextract and marker.is_file() and marker.read_text().strip() == sha:
        print(f'Already extracted: {relative}', flush=True)
        return
    with zipfile.ZipFile(path) as archive:
        members = [x for x in archive.infolist() if not x.is_dir()]
        expanded = sum(x.file_size for x in members)
        if shutil.disk_usage(root).free < expanded + 256 * 1024**2:
            raise RuntimeError(f'Need {expanded / 1e9:.2f} GB additional free space to extract {relative}')
        print(f'Extracting {relative}: {len(members)} files, {expanded / 1e9:.2f} GB', flush=True)
        for index, info in enumerate(members, 1):
            if stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError(f'Unexpected symlink: {info.filename}')
            destination = destination_for(info.filename, relative, root)
            if destination is None:
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            partial = destination.with_name(destination.name + '.dgm4-part')
            with archive.open(info) as source, partial.open('wb') as target:
                shutil.copyfileobj(source, target, 1024 * 1024)  # Includes ZIP CRC validation.
            os.replace(partial, destination)
            if index % 10000 == 0:
                print(f'  {index}/{len(members)} extracted', flush=True)
    marker.write_text(sha + '\n')


def validate(root):
    stats, missing = {}, []
    checked = set()
    for split in ['train', 'val', 'test']:
        path = root / 'DGM4/metadata' / f'{split}.json'
        with path.open() as f:
            rows = json.load(f)
        if not isinstance(rows, list) or not rows:
            raise ValueError(f'Invalid metadata: {path}')
        stats[split] = len(rows)
        for record in rows:
            image = record['image']
            parts = PurePosixPath(image).parts
            if not parts or parts[0] != 'DGM4' or '..' in parts or '\\' in image:
                raise ValueError(f'Unexpected metadata image path: {image}')
            if image in checked:
                continue
            checked.add(image)
            target = root / Path(*parts)
            if not target.is_file() or target.stat().st_size == 0:
                missing.append(image)
        print(f'{split}: {len(rows)} records checked', flush=True)
    stats['unique_image_paths'] = len(checked)
    stats['missing_images'] = len(missing)
    print(json.dumps(stats, indent=2), flush=True)
    if missing:
        raise RuntimeError(f'{len(missing)} missing/empty images, first 10: {missing[:10]}. '
                           'Rerun with --reextract if extraction markers already exist.')
    return stats


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', required=True, type=Path, help='Parent directory containing DGM4/')
    p.add_argument('--plan-only', action='store_true', help='Print files/size; do not download')
    p.add_argument('--verify-only', action='store_true', help='Check all metadata image paths; no downloads')
    p.add_argument('--reextract', action='store_true', help='Reextract all archives, replacing dataset files')
    args = p.parse_args()
    root = args.data_root.expanduser().resolve()
    if args.plan_only:
        print(json.dumps({'revision': REVISION, 'download_bytes': sum(x[1] for x in FILES),
                          'data_root': str(root), 'files': [x[0] for x in FILES]}, indent=2))
        return
    if int(os.environ.get('WORLD_SIZE', '1')) > 1:
        p.error('Run once with python, not with torchrun.')
    if args.verify_only:
        validate(root)
        return
    if not shutil.which('curl'):
        p.error('curl is required. Install it on the server before downloading.')
    root.mkdir(parents=True, exist_ok=True)
    cache = root / '.dgm4-downloads' / REVISION
    cache.mkdir(parents=True, exist_ok=True)
    # Prevent concurrent writers on Linux/macOS. A stale lock file is harmless.
    import fcntl
    with (cache / 'download.lock').open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            p.error('Another downloader is already using this data root.')
        for relative, size, sha in FILES:
            target = root / 'DGM4' / relative if relative.endswith('.json') else cache / relative
            download(relative, size, sha, target)
            if relative.endswith('.zip'):
                marker = cache / (relative.replace('/', '_') + '.extracted')
                unpack(target, relative, root, marker, sha, args.reextract)
        stats = validate(root)
        report = {'dataset': 'rshaojimmy/DGM4', 'revision': REVISION,
                  'data_root': str(root), 'splits': stats,
                  'source_files': [{'path': n, 'bytes': s, 'sha256': h} for n, s, h in FILES]}
        (root / 'DGM4/download_report.json').write_text(json.dumps(report, indent=2) + '\n')
        print(f'DGM4 ready. Use training argument: --data-root {root}', flush=True)


if __name__ == '__main__':
    main()
