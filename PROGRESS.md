# MSPS1 progress (agent memory; "continue from PROGRESS.md" must be enough)

## Current phase
Build done through E0 checks in progress. See checklist.

## Environment
- `.venv`: main env (Python 3.13.5; pip, no conda on this machine). `.venv-gpu`: JAX 0.12.0.1 CUDA 12 on GTX 1080 Ti.
- 12 threads (Ryzen 5 7600X), 30 GB RAM, 118 GB free disk at start.
- Run tests: `.venv/bin/python -m pytest -m "not slow and not campaign and not gpu"`; GPU tests: `.venv-gpu/bin/python -m pytest tests/test_gpu.py -m gpu`.

## Decisions and deviations
(see bottom sections)
