# Packaging verification

Checks completed locally on 2026-11-11. Python invoked through the existing
`work/hammer-apple/.venv-apple/bin/python` path, without resolving its symlink.
The local environment reports Python 3.10.7, torch 2.8.0, torchvision 0.23.0,
NumPy 1.26.4; CUDA runtime is absent and visible CUDA device count is zero.
This differs from the optional new CUDA environment's torch 2.5.1 pins.

- All 13 copied original files match source bytes and SHA-256 manifest entries.
- All 10 packaged Python files pass AST syntax parsing and module imports.
- All four shell scripts pass `bash -n`.
- The three requested portable unittest modules pass **14 tests**. They cover
  synthetic cache loading/corruption, frozen feature hooks, v1 train/resume/test,
  v2 parity/gradients/conditioning/penalty, and v2 train/resume/test.
- Wrapper dispatch checked with a recording Python substitute: help performs no
  work; all 11 action branches forward expected arguments; cached evaluation
  passes `--split test` without extraction; v1 uses lr `1e-5`; resume flags pass
  through; all four v2 output directory names and variant settings match the
  original launcher. Unknown actions/variants and missing required env fail.
- Setup default only prints help. Existing-venv overwrite is refused. Real
  `--check` passes all dependency imports and reports unavailable CUDA.
- Paths in packaged commands contain only environment-relative paths and the
  documented user's server defaults. No developer-local absolute paths added.

The initial extra `test_craf_residual.py` copy depended on the separate local
`fakenews` project; it was removed from the deliverable and manifest. The final
suite consists only of the three requested portable modules. An empty-array
compatibility issue found in the new wrapper on macOS Bash was fixed and the
complete dispatch checks passed afterward. Production sources were not edited.

Not executed: new-venv installation, network downloads, official-repo clone,
real DGM4 evaluation, frozen HAMMER extraction, CUDA training, or four-GPU runs.
No GPU throughput or benchmark metric is asserted by these packaging checks.
