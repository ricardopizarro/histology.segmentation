#!/bin/sh

# mv ~/ribbon.BIC_* ~/log_ribn
slice_nb_file='/data/shmuel/shmuel1/rap/histo/data/slice_nb.two.txt'
while read p; do
    for quad_nb in `seq 0 3`; do
        qsub -q all.q -l h_vmem=65G ribbon.BIC_CPU.test.sh $p $quad_nb
    done
done <$slice_nb_file
# qsub -q gpu.q -l h_vmem=100G ribbon.BIC_GPU.test.sh
# qsub -q gpu.q -l h_vmem=200G ribbon.BIC_GPU.sh
