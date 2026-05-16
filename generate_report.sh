#!/bin/bash

set -euo pipefail

LOG_FILE=~/qlib/logs/generate_report_$(date +%Y%m%d_%H%M%S).log

mkdir -p ~/qlib/logs

run() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] RUN: $*" | tee -a "$LOG_FILE"
    if ! "$@" >> "$LOG_FILE" 2>&1; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] FAILED: $*" | tee -a "$LOG_FILE"
        exit 1
    fi
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] OK: $*" | tee -a "$LOG_FILE"
}

cd ~/AI/akshare_data
run conda run -n quant python3 sync_multithread.py 2>&1
./syncToQlib.sh 2>&1

cd ~/qlib
run conda run -n quant python3 scripts/akshareToBin.py --max_workers 8 2>&1
run conda run -n quant python3 examples/workflow_dual_horizon.py 2>&1
run conda run -n quant python3 examples/view_exp_artifacts.py --send-email --exp-name 1D_Short_Term --action-plan --export-csv 2>&1
run conda run -n quant python3 examples/view_exp_artifacts.py --send-email --exp-name 5D_Mid_Term  --action-plan --export-csv 2>&1

echo "[$(date '+%Y-%m-%d %H:%M:%S')] All steps completed successfully." | tee -a "$LOG_FILE"
