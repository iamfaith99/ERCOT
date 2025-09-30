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
python -c "
import sys
# Import juliacall first to prevent warnings
try:
    import juliacall
    jl_version = juliacall.__version__
    print(f'   ✅ juliacall {jl_version}')
except ImportError:
    print(f'   ❌ juliacall not found')

# Then import other packages
packages = ['numpy', 'pandas', 'torch', 'stable_baselines3', 'pymc']
for pkg in packages:
    try:
        mod = __import__(pkg)
        version = getattr(mod, '__version__', 'unknown')
        print(f'   ✅ {pkg} {version}')
    except ImportError:
        print(f'   ❌ {pkg} not found')
"
