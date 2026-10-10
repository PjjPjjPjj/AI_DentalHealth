#!/usr/bin/env bash
# Data audit + missing resources report (Phase 1)
set -e
cd "$(dirname "$0")/.."
python -m src.utils.init
python -m src.data.audit
python -m src.data.missing_resources
echo "Audit done: outputs/reports/data_audit_report.json, missing_resources.md"
