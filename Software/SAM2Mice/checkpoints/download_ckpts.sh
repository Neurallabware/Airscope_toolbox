#!/usr/bin/env bash
# download.sh — Download a file from Google Drive using gdown

FILE_ID="1ixrVMJ512o_Zm4C6GXqqs40AErYCH0_6"
OUTPUT_FILE="SAM2_Mice_base_plus.pt"

# Check if gdown is installed
if ! command -v gdown &> /dev/null; then
  echo "Error: gdown is not installed. Please run 'pip install gdown' first."
  exit 1
fi

echo "Starting download from Google Drive..."
gdown --id "${FILE_ID}" -O "${OUTPUT_FILE}"

if [ $? -eq 0 ]; then
  echo "Download completed: ${OUTPUT_FILE}"
else
  echo "Download failed. Please check if the file is publicly shared, the link is correct, and your network connection is available."
  exit 1
fi

