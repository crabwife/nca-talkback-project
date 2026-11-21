#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
if [[ -n "${NCA_ENV_SCRIPT:-}" ]]; then
    source "$NCA_ENV_SCRIPT"
fi
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1
exec "${NCA_PYTHON:-python}" -u - "$@" <<'PYTHON'
import sys
from pathlib import Path
import nbformat
namespace = {'__name__': 'nca_pipeline'}
for code in nbformat.read(Path.cwd() / 'pipeline_tools.ipynb', 4).cells:
    exec(compile(code.source, 'pipeline_tools.ipynb', 'exec'), namespace)
namespace['main'](sys.argv[1:])
PYTHON
