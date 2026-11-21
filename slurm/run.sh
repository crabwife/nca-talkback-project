#!/usr/bin/env bash
set -euo pipefail
# Slurm runs a spool copy of this file; --chdir still points at the checkout.
launcher_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "$launcher_root/pipeline_tools.ipynb" ]]; then
    repo_root="$launcher_root"
elif [[ -f "$PWD/pipeline_tools.ipynb" ]]; then
    repo_root="$PWD"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "$SLURM_SUBMIT_DIR/pipeline_tools.ipynb" ]]; then
    repo_root="$SLURM_SUBMIT_DIR"
else
    printf 'Cannot locate pipeline_tools.ipynb; run from the repository root or submit with --chdir pointing there.\n' >&2
    exit 1
fi
cd "$repo_root"
if [[ -n "${NCA_ENV_SCRIPT:-}" ]]; then
    # Site login/module scripts may expand unset interactive-shell variables.
    set +u
    source "$NCA_ENV_SCRIPT"
    set -u
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
