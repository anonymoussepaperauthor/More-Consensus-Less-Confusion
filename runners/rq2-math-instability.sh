# !/bin/bash
find data/optimize/*/ -name "*.csv" | \
    parallel --jobs 50% --load 80% --progress \
    'echo "Running {/}..." && python RQ2/code/math-instability.py {} > RQ2/results/math-instability/{/}'
    