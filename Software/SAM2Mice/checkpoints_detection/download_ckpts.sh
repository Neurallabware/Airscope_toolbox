#!/usr/bin/env bash
# Download the YOLOv11 detector checkpoint from Google Drive using gdown.

set -euo pipefail

FILE_URL="https://drive.google.com/file/d/1JQXtWeko4inlEP-e9xruFqO7unlIpwYN/view?usp=drive_link"
OUTPUT_FILE="yolo11l_openfield_five_mice.pt"

if ! command -v gdown &> /dev/null; then
  echo "Error: gdown is not installed. Please run 'pip install gdown' first."
  exit 1
fi

echo "Downloading YOLOv11 detector checkpoint from Google Drive..."
gdown "${FILE_URL}" -O "${OUTPUT_FILE}"
echo "Download completed: ${OUTPUT_FILE}"
