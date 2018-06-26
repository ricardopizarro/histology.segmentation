#!/bin/sh

source /data/shmuel/shmuel1/rap/histo/venv_cpu/bin/activate

export TF_CPP_MIN_LOG_LEVEL=2

cd /data/shmuel/shmuel1/rap/histo/src
# python ribbon.inter-rater.py
slice_nb=$1
# python ribbon.test_convex_hull.py $slice_fn $segment_fn
# python ribbon.retile_quad.py $slice_nb 
python ribbon.retile_quad.py 0186 
