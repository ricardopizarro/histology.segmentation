import nibabel as nib
import numpy as np
import cv2 as cv
import glob
import json
import os,sys
import tensorflow as tf
from scipy import ndimage
import difflib
import random
from subprocess import check_call


def grab_files(path,end):
    return glob.glob(os.path.join(path,end))


def split_cross_valid(slices_fn,segments_fn):
    # slices_base=[os.path.basename(s) for s in slices_fn]
    # segments_base=[os.path.basename(s) for s in segments_fn]

    all_files =[]
    for n,seg_fn in enumerate(segments_fn):
        slice_fn=difflib.get_close_matches(seg_fn,slices_fn)[0]
        # slice_fn=[s for s in slices_fn if slice_base in s]
        if not slice_fn:
            print("Could not find an equivalent segment file {}".format(segment_fn))
            continue
        all_files.append((slice_fn,seg_fn))

    return all_files



slices_fn=[]
segments_fn=[]
slices=[]

data_path = ['/data/shmuel/shmuel1/deepthi/RM311_HighRes_Seg_Set1_1-70','/data/shmuel/shmuel1/deepthi/RM311_HighRes_Seg_Set2_71-113']
for p in data_path:
    ss=grab_files(p,'*')
    slices+=[os.path.basename(s) for s in ss]
    slices_fn += grab_files(p,'*/*jpg.nii')
    segments_fn += grab_files(p,'*/*segmented.nii*')

# print(segments_fn)

all_files = split_cross_valid(slices_fn,segments_fn)

print(all_files)

fn='/data/shmuel/shmuel1/rap/histo/data/segmented_files.txt'
thefile = open(fn,'w')

for item in all_files:
    thefile.write('%s\n' % item)


