#!/usr/bin/env bash
# Convert all raw datasets to processed YOLO layouts (Phase 2-5 conversions)
set -e
cd "$(dirname "$0")/.."
python -m src.data.convert_caries6313
python -m src.data.convert_oral_mamba
python -m src.data.convert_airc
python -m src.data.convert_io150k
echo "Prepare done. Check datasets/processed/ and metadata split_manifest.csv"
