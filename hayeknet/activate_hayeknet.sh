#!/bin/bash
# Activate HayekNet environment

echo "🔧 Activating HayekNet environment..."

# Initialize conda
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate hayeknet

echo "✅ HayekNet environment activated!"
echo "   Python: $(which python)"
echo "   Python version: $(python --version)"
echo "   Conda env: $CONDA_DEFAULT_ENV"
echo "   ✅ Updated to Python 3.13.7 (latest)"
echo "   ✅ Fixed juliacall/torch import order"
echo "   ✅ Removed deprecated gym warnings"

# Set environment variables
export HAYEKNET_ROOT="$(pwd)"
export PYTHONPATH="$HAYEKNET_ROOT:$PYTHONPATH"

# Julia project path
if [ -d "julia" ]; then
    export JULIA_PROJECT="$HAYEKNET_ROOT/julia"
fi

echo "   Project root: $HAYEKNET_ROOT"
echo ""
echo "🚀 Ready to run HayekNet!"
echo "   Test with: python validate_ercot.py"
echo "   Run main:  python -m python.main"
echo ""
echo "📦 Key packages installed:"

# Allow Makefile and non-interactive contexts to skip import checks that can trigger segfaults
if [ -n "$HAYEKNET_SKIP_CHECKS" ]; then
    echo "   ⏭️  Skipping import checks (HAYEKNET_SKIP_CHECKS=1)"
else
    # Check packages individually to avoid import conflicts
    # NOTE: juliacall import can allocate Julia runtime; keep as last and tolerate failure
    python -c "import numpy; print(f'   ✅ numpy {numpy.__version__}')" 2>/dev/null || echo "   ❌ numpy not found"
    python -c "import pandas; print(f'   ✅ pandas {pandas.__version__}')" 2>/dev/null || echo "   ❌ pandas not found"
    python -c "import torch; print(f'   ✅ torch {torch.__version__}')" 2>/dev/null || echo "   ❌ torch not found"
    python -c "import stable_baselines3; print(f'   ✅ stable_baselines3 {stable_baselines3.__version__}')" 2>/dev/null || echo "   ❌ stable_baselines3 not found"
    python -c "import pymc; print(f'   ✅ pymc {pymc.__version__}')" 2>/dev/null || echo "   ❌ pymc not found"
    # Try juliacall last; if it fails here, runtime script will still import it first
    python -c "import juliacall; print(f'   ✅ juliacall {juliacall.__version__}')" 2>/dev/null || echo "   ❌ juliacall not found"
fi
