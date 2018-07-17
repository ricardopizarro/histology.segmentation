#!/bin/bash
sets=/data/shmuel/shmuel1/rap/histo/data/rm311
rm -rf $sets/*
root=/data/shmuel/shmuel1/deepthi/
declare -a arr=("RM311_HighRes_Seg_Set1_1-70/*" "RM311_HighRes_Seg_Set2_71-113/*" "RM311_HighRes_Seg_Set3_1134-132/*" "RM311_HighRes_Seg_Set4_133-180/*")
for set in "${arr[@]}" ; do
    set_dir=$root$set
    # echo $set_dir
    for d in $set_dir ; do
        ln -s $d $sets/$(basename $d)
    done
done



