#!/usr/bin/env bash
# download.sh — Download a file from Google Drive using gdown

FILE_ID="1JQXtWeko4inlEP-e9xruFqO7unlIpwYN"
OUTPUT_FILE="yolo11l_openfield_five_mice.pt"

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
