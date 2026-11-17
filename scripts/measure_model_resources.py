#!/usr/bin/env python3
"""Measure exact additional head parameters/FP32 weight bytes, not runtime VRAM."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'gpu'))
from craf_residual import FrozenCRAFResidual as V1
from craf_residual_v2 import FrozenCRAFResidual as V2


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--check-only', action='store_true')
    args = p.parse_args()
    rows = []
    for name, model in [('v1', V1()), ('v2_control', V2()),
                        ('v2_decision', V2(condition_on_decision=True))]:
        n = sum(v.numel() for v in model.parameters() if v.requires_grad)
        rows.append(dict(model=name, feature_dim=768, hidden_dim=256,
                         trainable_parameters=n, fp32_weight_bytes=n * 4))
    result = dict(measurement='exact parameter count from packaged architecture; no GPU timing/memory measurement', rows=rows)
    path = ROOT / 'results/model_resources.json'
    if args.check_only:
        if json.loads(path.read_text()) != result:
            raise ValueError('Resource counts differ from saved artifact')
    else:
        path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
