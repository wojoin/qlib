#!/bin/bash

set -euo pipefail

LOG_FILE=~/ai/qlib/logs/generate_report_$(date +%Y%m%d_%H%M%S).log

mkdir -p ~/ai/qlib/logs

run() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] RUN: $*" | tee -a "$LOG_FILE"
    if ! "$@" 2>&1 | tee -a "$LOG_FILE"; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] FAILED: $*" | tee -a "$LOG_FILE"
        exit 1
    fi
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] OK: $*" | tee -a "$LOG_FILE"
}

cd ~/ai/akshare_data
run conda run --no-capture-output -n quant python3 sync_multithread.py
run ./syncToQlib.sh

cd ~/ai/qlib
run conda run --no-capture-output -n quant python3 scripts/akshareToBin.py --max_workers 8
run conda run --no-capture-output -n quant python3 examples/workflow_dual_horizon_pro2.py
run conda run --no-capture-output -n quant python3 examples/view_exp_artifacts.py --send-email --exp-name 1D_Short_Term --action-plan --export-csv
run conda run --no-capture-output -n quant python3 examples/view_exp_artifacts.py --send-email --exp-name 5D_Mid_Term --action-plan --export-csv
run conda run --no-capture-output -n quant python3 examples/view_exp_artifacts.py --send-email --exp-name Adaptive_Exit --action-plan --export-csv

echo "[$(date '+%Y-%m-%d %H:%M:%S')] All steps completed successfully." | tee -a "$LOG_FILE"
