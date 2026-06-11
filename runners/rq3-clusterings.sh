# !/bin/bash
find data/optimize/*/ -name "*.csv" | \
    parallel --jobs 50% --load 80% --progress \
    'echo "Running {/}..." && python RQ3/code/clusterings.py {} > RQ3/results/clusterings/{/}'
