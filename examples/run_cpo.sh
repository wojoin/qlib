#!/bin/shell

python examples/workflow_by_cpo.py 2>&1 | tee examples/logs/full_$(date +%Y%m%d_%H%M%S).log

python examples/view_cpo_artifacts.py --action-plan --export-csv