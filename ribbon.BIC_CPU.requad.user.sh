#!/bin/sh

source /data/shmuel/shmuel1/rap/histo/venv_cpu/bin/activate
cd /data/shmuel/shmuel1/rap/histo/src

data_source=$1
data_target=$2

python ribbon.retile_quad.user.py $data_source $data_target

