import nibabel as nib
import numpy as np
import glob
import json
import os
import sys
import difflib

def grab_files(path,end):
    return glob.glob(path+end)

def consolidate_seg(seg):
    # swap elements labeled 6 and 2 to 0.  
    # elements labeled 6 indicate a tear in the white matter
    # elements labeled 5 indicate a fold in the gray matter
    # elements labeled 4 indicate subcortex or cerebellum
    # elements labeled 2 indicate a tear in the gray matter
    # elements labeled 3 indicate a blood vessel in the gray matter
    for m,row in enumerate(seg):
        for n,elem in enumerate(row):
            if elem in [3,5]:
		seg[m][n]=1 # make blood vessels as gray matter
            elif elem in [2,4,6]:
                seg[m][n]=0
    return seg

def segment(tile,seg):
    # We used the labeled seg to segment the subcortex and cerebellum
    # To mask this portion out we simply make it white like the background [255,255,255]
    for m,row in enumerate(seg):
        for n,elem in enumerate(row):
            if elem in [4]:
                tile[m,n,:]=10
            # elif 0 in hull[m][m]:
            #     tile[m,n,:]=20
    return tile



def get_channel(img,fn):
    ch_ret=-1
    num_ch_labeled=0
    for ch in range(img.shape[2]):
        # print(ch)
        # print(np.unique(img[:,:,ch]))
        if len(np.unique(img[:,:,ch]))>1:
            ch_ret=ch
            num_ch_labeled+=1
    if ch_ret<0:
        print("Warning! No labels detected : {}".format(fn))
        return np.asarray(0)
    elif num_ch_labeled>1:
        print('Warning! Multiples channels were labeled : {}'.format(fn))
        print('We will use channel 2')
        ch_ret=1

    return img[:,:,ch_ret]


def dice_coef(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def get_data_shape(fn):
    img = nib.load(fn)
    return img.get_data(),img.shape

def calc_dc(data):
    y = np.bincount(data)
    ii = np.nonzero(y)[0]
    # tn = true negative, means the two raters agree it is not gray matter
    # fn = false negative, means jeeda marked it as not gray matter and deepthi marked as gray matter
    # fp = false positive, means jeeda marked it as gray matter and deepthi marked as not gray matter
    # tp = true positive,  means the two raters agree it is gray matter
    ii_str=['tn','fn','fp','tp']
    tn,fn,fp,tp = y
    print(zip(ii_str,y[ii]))
    return 2.0*tp/(2*tp+fp+fn)

def compute_dc(fn1,fn2):
    print('{} : {}'.format(fn1,fn2))

    data1,shape1=get_data_shape(fn1)
    data2,shape2=get_data_shape(fn2)

    if not shape1==shape2:
        print(shape1,shape2)
        print('Warning: These two segmentation files are not identical in size')
        sys.exit()

    data1=get_channel(data1,fn1)
    print(np.unique(data2))
    data2=get_channel(data2,fn2)
    if not data1.any() or not data2.any():
        return 0.0

    data1=np.asarray(consolidate_seg(data1.tolist()))
    data2=np.asarray(consolidate_seg(data2.tolist()))
    
    print(data1.shape,data2.shape)
    print(np.unique(data1),np.unique(data2))

    data12 = data1+2*data2
    # dc=calc_dc(data12.flatten())
    # sys.exit()
    return calc_dc(data12.flatten())



path_001 = '/data/shmuel/shmuel1/deepthi/RM311_HighRes_Seg_Set2_71-113/'
if not os.access(path_001, os.R_OK):
    print('Cannot read any of the files in {}'.format(path_001))
    sys.exit()

segments_001=glob.glob(path_001+'*/*segmented.nii*')
segments_001=[os.path.basename(seg) for seg in segments_001]

path_002 = '/data/shmuel/shmuel1/mok/histology_nhp/segmentation/transfer_to_jeedha/segmented/'
if not os.access(path_002, os.R_OK):
    print('Cannot read any of the files in {}'.format(path_002))
    sys.exit()

segments_002=glob.glob(path_002+'*')
segments_002=sorted([os.path.basename(seg) for seg in segments_002])


segments_001=[difflib.get_close_matches(seg,segments_001)[0] for seg in segments_002]

segments_001_002=zip(segments_001,segments_002)
# print(segments_001_002)


print('\nComputing the inter rater coefficient, using Sorensen index as:')
print('tn=true negative : the two raters agree it is not gray matter')
print('fn=false negative: Jeeda marked as not gray matter and Deepthi marked as gray matter')
print('fp=false positive: Jeeda marked as gray matter and Deepthi marked as not gray matter')
print('tp=true positive : the two raters agree it is gray matter\n')

dice=[]
for seg in segments_001_002:
    slice_dir=seg[0][:4]
    dc=compute_dc(os.path.join(path_001,slice_dir,seg[0]),os.path.join(path_002,seg[1]))
    print('dice : {0:0.3f}'.format(dc))
    # sys.exit()
    dice.append(dc)

print(sorted(dice))
dice=np.asarray(dice)
print('\n>>>Dice coefficient<<< {0} quadrants : {1:0.3f} +/- {2:0.3f}\n'.format(len(dice),np.mean(dice),np.std(dice)))




