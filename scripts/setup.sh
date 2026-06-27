#!/usr/bin/env bash
# CS336 setup script — run from the project root
set -euo pipefail

echo "==> Installing uv (if needed)..."
if ! command -v uv &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

echo "==> Syncing dependencies..."
uv sync --group dev

echo "==> Registering Jupyter kernel..."
source .venv/bin/activate
python -m ipykernel install --user --name cs336 --display-name "Python (cs336)"

echo ""
echo "✅ Done. Activate with:  source .venv/bin/activate"
echo "   Launch Jupyter:       jupyter lab"
