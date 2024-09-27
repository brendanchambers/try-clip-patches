#!/bin/bash

# quick setup for local machine, todo setup for replicability
# reuse existing conda env for quick first pass


# activate conda env
source ../set_bash_env.sh
pushd $project_dir
echo "working directory: $project_dir"

# sanity check clip model
conda run -n pytorch-env --live-stream python src/explore/explore.py