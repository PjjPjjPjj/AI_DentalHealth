#!/usr/bin/env bash
# Single-image demo
set -e
cd "$(dirname "$0")/.."
python demo.py --image demo_images/test.jpg "$@"
