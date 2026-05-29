#!/usr/bin/env bash
# Download the SAM2Mice checkpoint from Google Drive using gdown.

set -euo pipefail

FILE_URL="https://drive.google.com/file/d/1aTYiL1qt23vUAGbHvh2Yrk3r16i95bnU/view?usp=drive_link"
OUTPUT_FILE="SAM2_Mice_base_plus.pt"

if ! command -v gdown &> /dev/null; then
  echo "Error: gdown is not installed. Please run 'pip install gdown' first."
  exit 1
fi

echo "Downloading SAM2Mice checkpoint from Google Drive..."
gdown "${FILE_URL}" -O "${OUTPUT_FILE}"
echo "Download completed: ${OUTPUT_FILE}"
