#!/bin/sh

mv ~/ribbon.BIC_* ~/log_ribn
segment_files='/data/shmuel/shmuel1/rap/histo/data/two_segmented_hull_files.txt'
while IFS=" " read -r slice_fn segment_fn hull_fn remainder; do
    qsub -q all.q -l h_vmem=65G ribbon.BIC_CPU.test.sh $slice_fn $segment_fn $hull_fn
done <$segment_files
# qsub -q gpu.q -l h_vmem=100G ribbon.BIC_GPU.test.sh
# qsub -q gpu.q -l h_vmem=200G ribbon.BIC_GPU.sh
