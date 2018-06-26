#!/bin/sh

mv ~/ribbon.BIC_* ~/log_ribn
qsub -q all.q -l h_vmem=20G ribbon.BIC_CPU.requad.sh 0186

# slice_nb='/data/shmuel/shmuel1/rap/histo/data/txt_files/slice_nb_seg.two.txt'
# while IFS=" " read -r slice_nb remainder; do
#     qsub -q all.q -l h_vmem=20G ribbon.BIC_CPU.requad.sh $slice_nb
# done <$slice_nb
# qsub -q gpu.q -l h_vmem=100G ribbon.BIC_GPU.test.sh
# qsub -q gpu.q -l h_vmem=200G ribbon.BIC_GPU.sh
