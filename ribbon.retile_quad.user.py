import nibabel as nib
import numpy as np
import glob
import sys
import os
import difflib
import scipy.ndimage
from subprocess import check_call

def grab_files(path,end):
    return glob.glob(os.path.join(path,end))

def combine_files(slices_fn,segments_fn):
    # print(sorted(slices_fn))
    files =[] 
    for n,segment_fn in enumerate(segments_fn):
        slice_fn=difflib.get_close_matches(segment_fn,slices_fn)[0]
        if not slice_fn:
            print("Could not find an equivalent segment file {}".format(segment_fn))
            continue
        files.append((slice_fn,segment_fn))
    return files

def requad_files(s_files,out_dir):

    slices,segments=map(list, zip(*s_files))
    requad(slices,'slice',out_dir)
    requad(segments,'segmented',out_dir)

    return

def get_channel(img):
    ch_ret=-1
    num_ch_labeled=0
    for ch in range(img.shape[2]):
        # print(ch)
        # print(np.unique(img[:,:,ch]))
        if len(np.unique(img[:,:,ch]))>1:
            ch_ret=ch
            num_ch_labeled+=1
    return ch_ret,num_ch_labeled

def get_order(ch):
    if ch==0:
        return [1,0,2]
    elif ch==1:
        return [0,1,2]
    elif ch==2:
        return [0,2,1]

def reorder(q,quad_fn):
    ch,num_ch_labeled=get_channel(q)
    if ch<0:
        print("{} no pink labels".format(quad_fn))
    elif num_ch_labeled>1:
        print("{} has multiple channels with labels ch {} => 1".format(quad_fn,ch))
        ch=1
    new_order=get_order(ch)
    return q[:,:,new_order]

def requad(quad_four,file_end,out_dir):

    q0=nib.load(quad_four[0]).get_data()
    q1=nib.load(quad_four[1]).get_data()
    q2=nib.load(quad_four[2]).get_data()
    q3=nib.load(quad_four[3]).get_data()
    if 'seg' in file_end:
        q0 = reorder(q0,quad_four[0])
        q1 = reorder(q1,quad_four[1])
        q2 = reorder(q2,quad_four[2])
        q3 = reorder(q3,quad_four[3])
    print(quad_four[0])
    print(q0.shape)

    print(quad_four[1])
    print(q1.shape)

    col0=np.concatenate((q0,q1),axis=1)
    print(col0.shape)

    print(quad_four[2])
    print(q2.shape)

    print(quad_four[3])
    print(q3.shape)

    col1=np.concatenate((q2,q3),axis=1)
    print(col1.shape)

    data=np.concatenate((col0,col1),axis=0)
    print(data.shape)
    ratio=[0.1,0.1,1.0]
    data=scipy.ndimage.zoom(data,ratio)
    print(data.shape)
    data=np.reshape(data,data.shape+(1,))

    # out_dir='/data/shmuel/shmuel1/rap/histo/data/rm311_requad'
    anchor='whole'
    fn_parts=os.path.basename(quad_four[0]).split(anchor)
    fn=os.path.join(out_dir,fn_parts[0]+anchor+'.'+file_end+'.nii')

    # print(fn)
    save_to_nii(data,fn)



def save_to_nii(data,fn):
    affine=np.eye(len(data.shape))
    img = nib.Nifti1Image(data,affine)
    path=fn
    print(path)
    if not os.path.isfile(path+'.gz'):
        nib.save(img,path)
        check_call(['gzip', path])
    else:
        print('File {} already exists'.format(path))



# Remember that sys.argv[0] is the name of the script.

# input dir where the hi-res quad files are located
data_source=sys.argv[1]
# output dir where the lo-res slice will be saved located
data_target=sys.argv[2]

if not os.access(data_source, os.R_OK):
    print('Cannot read any of the files in {}'.format(data_source))
    sys.exit()

if not os.access(data_target, os.W_OK):
    print('Cannot read any of the files in {}'.format(data_target))
    sys.exit()

slices_fn = grab_files(data_source,'*.jpg.nii')
segments_fn = grab_files(data_source,'*segmented.nii*')
files = combine_files(slices_fn,segments_fn)

requad_files(files,data_target)


