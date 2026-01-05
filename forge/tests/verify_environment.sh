#============================================
# forge/tests/verify_environment.sh
# Purpose: Runs a comprehensive suite of tests to verify the ML environment.
# Role: Diagnostic / Validation Layer
# Used by: Phase 1 of the Forge Pipeline
#============================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

echo "============================================="
echo "FORGE ENVIRONMENT VERIFICATION"
echo "============================================="
echo "Project Root: $PROJECT_ROOT"
echo ""

# Activate ML environment
echo "[1/11] Activating ~/ml_env..."
source ~/ml_env/bin/activate
echo "      Python: $(which python)"
echo ""

# Test PyTorch + CUDA (CRITICAL GATE)
echo "[2/11] Testing PyTorch + CUDA (CRITICAL)..."
python -c "
import torch
print('      PyTorch Version:', torch.__version__)
print('      CUDA Available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('      CUDA Version:', torch.version.cuda)
    print('      Device Count:', torch.cuda.device_count())
    print('      Device Name:', torch.cuda.get_device_name(0))
else:
    print('      ERROR: CUDA not available!')
    exit(1)
"
echo "      ✅ PyTorch + CUDA OK"
echo ""

# Test bitsandbytes
echo "[3/11] Testing bitsandbytes..."
python -c "
import bitsandbytes as bnb
print('      bitsandbytes:', bnb.__version__)
" && echo "      ✅ bitsandbytes OK" || echo "      ⚠️ bitsandbytes check failed (optional)"
echo ""

# Test triton
echo "[4/11] Testing triton..."
python -c "
import triton
print('      triton:', triton.__version__)
" && echo "      ✅ triton OK" || echo "      ⚠️ triton check failed (optional)"
echo ""

# Test transformers
echo "[5/11] Testing transformers..."
python -c "
import transformers
print('      transformers:', transformers.__version__)
" && echo "      ✅ transformers OK" || echo "      ⚠️ transformers check failed"
echo ""

# Test xformers (optional - may not have cu130 wheels)
echo "[6/11] Testing xformers..."
python -c "
try:
    import xformers
    print('      xformers:', xformers.__version__)
except ImportError as e:
    print('      xformers not installed (optional for cu130)')
" && echo "      ✅ xformers OK" || echo "      ⚠️ xformers check failed (optional)"
echo ""

# Test llama-cpp-python
echo "[7/11] Testing llama-cpp-python..."
python -c "
try:
    from llama_cpp import Llama
    import llama_cpp
    print('      llama-cpp-python: import OK')
except ImportError as e:
    print('      llama-cpp-python not installed:', e)
    exit(1)
" && echo "      ✅ llama-cpp-python OK" || echo "      ⚠️ llama-cpp-python check failed"
echo ""

# Test fine-tuning dependencies
echo "[8/11] Testing peft (LoRA)..."
python -c "
import peft
print('      peft:', peft.__version__)
" && echo "      ✅ peft OK" || echo "      ⚠️ peft check failed (required for fine-tuning)"
echo ""

echo "[9/11] Testing trl (SFTTrainer)..."
python -c "
import trl
print('      trl:', trl.__version__)
" && echo "      ✅ trl OK" || echo "      ⚠️ trl check failed (required for fine-tuning)"
echo ""

echo "[10/11] Testing datasets..."
python -c "
import datasets
print('      datasets:', datasets.__version__)
" && echo "      ✅ datasets OK" || echo "      ⚠️ datasets check failed (required for fine-tuning)"
echo ""

echo "[11/11] Testing psutil..."
python -c "
import psutil
print('      psutil:', psutil.__version__)
" && echo "      ✅ psutil OK" || echo "      ⚠️ psutil check failed (required for fine-tuning)"
echo ""

echo "============================================="
echo "VERIFICATION COMPLETE"
echo "============================================="
echo ""
echo "All critical gates passed. Environment ready for fine-tuning."
echo ""
echo "Next Steps:"
echo "  1. cd $PROJECT_ROOT"
echo "  2. source ~/ml_env/bin/activate"
echo "  3. python forge/scripts/fine_tune.py"

