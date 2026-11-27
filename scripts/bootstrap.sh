#!/usr/bin/env bash
# One-time local setup: virtual environment, dependencies, Git + DVC initialisation,
# git hooks and a first commit. Safe to re-run: existing steps are skipped.
#
# Usage (from the repository root):
#   bash scripts/bootstrap.sh
#   bash scripts/bootstrap.sh --dvc-remote https://dagshub.com/<user>/brain-tumor-mlops.dvc
set -euo pipefail

DVC_REMOTE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dvc-remote) DVC_REMOTE="$2"; shift 2 ;;
    -h|--help) sed -n '2,8p' "$0"; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

cd "$(dirname "$0")/.."
step() { printf '\n\033[1;34m==> %s\033[0m\n' "$1"; }

step "Python virtual environment (.venv)"
if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --quiet --upgrade pip

if ! python -c "import torch" >/dev/null 2>&1; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    step "Installing PyTorch (CUDA build)"
    python -m pip install --quiet torch torchvision
  else
    # No NVIDIA GPU: the CPU build is ~10x smaller than the default CUDA wheels.
    step "Installing PyTorch (CPU build: no NVIDIA GPU detected, training runs on Colab)"
    python -m pip install --quiet torch torchvision --index-url https://download.pytorch.org/whl/cpu
  fi
fi

step "Installing the package and dev tools (this can take a few minutes)"
python -m pip install --quiet -e . -r requirements-dev.txt -r requirements-api.txt

step "Git repository"
if [[ ! -d .git ]]; then
  git init -b main
fi

step "DVC"
if [[ ! -d .dvc ]]; then
  dvc init
fi
if [[ -n "$DVC_REMOTE" ]]; then
  dvc remote add -d -f origin "$DVC_REMOTE"
  echo "Now store credentials locally (never committed):"
  echo "  dvc remote modify origin --local auth basic"
  echo "  dvc remote modify origin --local user <user>"
  echo "  dvc remote modify origin --local password <token>"
fi

step "Secrets file"
if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "Created .env from .env.example: fill in your DagsHub credentials."
fi

step "Git hooks (pre-commit)"
pre-commit install

step "Running the test suite"
python -m pytest -q -p no:warnings

step "First commit"
if ! git rev-parse --verify HEAD >/dev/null 2>&1; then
  git add -A
  git commit -m "chore: initial project structure" --no-verify
else
  echo "Repository already has commits: skipped."
fi

cat <<'EOF'

Setup complete. Next steps:
  1. Create the GitHub repository, then:
       git remote add origin https://github.com/<user>/brain-tumor-mlops.git
       git push -u origin main
  2. Connect it on DagsHub and fill in .env (see docs/mlops_setup.md).
  3. Build the dataset:  dvc repro   then   dvc push
EOF
