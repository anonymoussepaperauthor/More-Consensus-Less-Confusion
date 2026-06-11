# !/bin/bash
find data/optimize/*/ -name "*.csv" | \
    parallel --jobs 50% --load 80% --progress \
    'echo "Running {/}..." && python RQ2/code/model-complexity.py {} > RQ2/results/model-complexity/{/}'
    