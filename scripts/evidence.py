"""Resolve original evidence names to the single stored copy."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def evidence_path(relative, root=ROOT):
    index = json.loads((root / 'results/SERVER_EVIDENCE_INDEX.json').read_text())
    for item in index['files']:
        if item['source'] == str(relative):
            return root / item['archived']
    raise FileNotFoundError(f'Unindexed evidence: {relative}')


def normalized_sha(value):
    payload = json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def transcription_record(filename, root=ROOT):
    """Return the recorded fingerprint, without inventing a second source copy."""
    records = json.loads((root / 'results/user_reported/TRANSCRIPTIONS.json').read_text())
    return records['files'][filename]
