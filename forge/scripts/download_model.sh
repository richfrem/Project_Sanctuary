#!/bin/bash
# ==============================================================================
# DOWNLOAD_MODEL.SH (v1.1)
#
# This script downloads the base pre-trained model from Hugging Face.
# It is idempotent, meaning it will not re-download the model if it already
# exists in the target directory.
#
# It requires a Hugging Face token for authentication, which should be stored
# in a .env file at the project root.
# ==============================================================================

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
MODEL_ID="Qwen/Qwen2-7B-Instruct"

# --- Determine Project Root and Paths ---
# This finds the script's own directory, then navigates to the forge root.
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
FORGE_ROOT="$SCRIPT_DIR/.."
PROJECT_ROOT="$FORGE_ROOT/../.."
OUTPUT_DIR="$FORGE_ROOT/models/base/$MODEL_ID" # CORRECTED PATH
ENV_FILE="$PROJECT_ROOT/.env"

echo "--- 🔽 Model Downloader Initialized ---"
echo "Model to download:  $MODEL_ID"
echo "Target directory:   $OUTPUT_DIR"
echo "========================================="

# --- Check if Model Already Exists ---
if [ -d "$OUTPUT_DIR" ] && [ "$(ls -A "$OUTPUT_DIR")" ]; then
  echo "✅ Model already exists locally. Skipping download."
  echo "========================================="
  exit 0
fi

echo "Model not found locally. Preparing to download..."
mkdir -p "$OUTPUT_DIR"

# --- Load Hugging Face Token ---
if [ ! -f "$ENV_FILE" ]; then
  echo "🛑 CRITICAL: '.env' file not found in the project root."
  echo "Please create a file named '.env' in the main Project_Sanctuary directory with the following content:"
  echo "HUGGING_FACE_TOKEN='your_hf_token_here'"
  exit 1
fi

# Extract token, removing potential Windows carriage returns and whitespace
HF_TOKEN=$(grep HUGGING_FACE_TOKEN "$ENV_FILE" | cut -d '=' -f2 | tr -d '[:space:]' | tr -d "'\"")

if [ -z "$HF_TOKEN" ] || [ "$HF_TOKEN" = "your_hf_token_here" ]; then
  echo "🛑 CRITICAL: HUGGING_FACE_TOKEN is not set in your .env file."
  echo "Please get a token from https://huggingface.co/settings/tokens and add it to your .env file."
  exit 1
fi

echo "🔐 Hugging Face token loaded successfully."

# --- Execute Download ---
echo "⏳ Starting download from Hugging Face Hub. This will take several minutes..."
echo "(Approx. 15 GB, depending on your connection speed)"

# Use a Python one-liner with the huggingface_hub library to perform the download
# We pass the shell variables as arguments to the python command
python3 -c "
from huggingface_hub import snapshot_download
import sys

# Get arguments passed from the shell
repo_id = sys.argv[1]
local_dir = sys.argv[2]
token = sys.argv[3]

print(f'Downloading {repo_id}...')
snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,
    token=token,
    local_dir_use_symlinks=False # Use direct copies to avoid WSL symlink issues
)
print('Download complete.')
" "$MODEL_ID" "$OUTPUT_DIR" "$HF_TOKEN"


echo "========================================="
echo "🏆 SUCCESS: Base model downloaded to '$OUTPUT_DIR'."
echo "You are now ready to run the fine-tuning script."
echo "--- 🔽 Model Downloader Complete ---"