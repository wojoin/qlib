#!/bin/bash

# 目标目录
QLIB_DIR="/Users/joseph/qlib"

cd "$QLIB_DIR" || exit
conda run -n quant python scripts/akshareToBin.py --src examples/data/20260512 --max_workers 8

