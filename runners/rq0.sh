# !/bin/bash
find data/optimize/*/ -name "*.csv" | \
    parallel --jobs 50% --load 80% --progress \
    'echo "Running {/}..." && python RQ0/code/sensitivity_analysis.py {} > RQ0/results/{/}'
    