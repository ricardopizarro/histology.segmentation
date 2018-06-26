import nibabel as nib
import numpy as np
import glob
import sys
import os
import difflib
import scipy.ndimage
from subprocess import check_call
import random
from PIL import Image
import png
from scipy import stats

def grab_files(path,end):
    return glob.glob(os.path.join(path,end))

def split_train_valid(slices_fn,segments_fn):
    # print(sorted(slices_fn))
    files =[] 
    for n,segment_fn in enumerate(segments_fn):
        # slice_quad_number = os.path.basename(segment_fn).split('segmented.nii')[0].upper()
        # print(os.path.basename(segment_fn))
        slice_fn=difflib.get_close_matches(segment_fn,slices_fn)[0]
        # slice_fn=[fn for fn in slices_fn if slice_quad_number in fn.upper()]
        if not slice_fn:
            print("Could not find an equivalent segment file {}".format(segment_fn))
            continue
        # print(slice_fn)
        files.append((slice_fn,segment_fn))
    return files



def get_slices(files):
    slices=[]
    for f in files:
        slices+=[os.path.basename(f[0])[:4]]
    return list(set(slices))

def range_RGB(img):
    return (img-np.amin(img))*(255.0/(np.ptp(img)))


def rgb_2_lum(img):
    # the rgb channel is located at axis=2 for the data
    img=0.2126*img[:,:,0]+0.7152*img[:,:,1]+0.0722*img[:,:,2]
    img=range_RGB(img)
    return img


def RGB_seg(seg_data):
    # seg_RGB=np.zeros(seg_data.shape+(3,))
    seg_RGB=np.zeros(seg_data.shape+(3,))
    seg_RGB[:] = np.nan
    seg_list=seg_data.tolist()
    for m,row in enumerate(seg_list):
        for n,elem in enumerate(row):
            if elem in [1]:
                seg_RGB[m,n,:]=np.array([1,0,0]) # red
            elif elem in [2]:
                seg_RGB[m,n,:]=np.array([0,1,0]) # green
            elif elem in [3,8]:
                seg_RGB[m,n,:]=np.array([0,0,1]) # blue
            elif elem in [4]:
                seg_RGB[m,n,:]=np.array([1,1,0]) # yellow
            elif elem in [5]:
                seg_RGB[m,n,:]=np.array([0,1,1]) # cyan
            elif elem in [6]:
                seg_RGB[m,n,:]=np.array([1,0,1]) # pink
            elif elem in [7]:
                seg_RGB[m,n,:]=np.array([1,1,1]) # pale
    return 255*seg_RGB

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

def consolidate_seg(seg):
    # swap elements labeled 6 and 2 to 0.  
    # elements labeled 6 indicate a tear in the white matter
    # elements labeled 5 indicate a fold in the gray matter
    # elements labeled 4 indicate subcortex or cerebellum
    # elements labeled 2 indicate a tear in the gray matter
    # elements labeled 3 indicate a blood vessel in the gray matter
    for m,row in enumerate(seg):
        for n,elem in enumerate(row):
            if elem in [3]:
                seg[m][n]=1
            elif elem in [2,4,5,6,7,8]:
                seg[m][n]=0
    return seg

def segment(tile,seg,white):
    # We used the labeled seg to segment the subcortex and cerebellum
    # To mask this portion out we simply make it a high value of 10
    for m,row in enumerate(seg):
        for n,elem in enumerate(row):
            if elem in [2,4,5,6,7]:
                tile[m,n,:]=white
            # elif 0 in hull[m][m]:
            #     tile[m,n,:]=20
    return tile

def flip_img(tile,seg,f1,f2):
    # tile is (2560,2560)
    # seg is (2560,2560)
    # f1=int(2*np.random.randint(0,2)-1)
    # f2=int(2*np.random.randint(0,2)-1)
    return tile[::f1,::f2,...],seg[::f1,::f2,...]

def gen_png(slice_fn,segment_fn,f):
    out_dir = '/data/shmuel/shmuel1/rap/histo/data/rm311_requad/pdf/'

    seg_data=nib.load(segment_fn).get_data()
    ch,num_ch_labeled=get_channel(seg_data)
    if ch<0:
        print("{} does not have pink labels".format(segment_fn))
    elif num_ch_labeled>1:
        print("{} has multiple channels with labels ch {} => 1".format(segment_fn,ch))
        ch=1
    seg_data=np.squeeze(seg_data[:,:,ch]).tolist()

    slice_RGB = np.squeeze(nib.load(slice_fn).get_data())
    slice_RGB = range_RGB(slice_RGB)
    white= int(stats.mode(slice_RGB, axis=None)[0])
    slice_RGB = segment(slice_RGB,seg_data,white)
    print(slice_RGB.shape,np.amax(slice_RGB),np.amin(slice_RGB))

    seg_data=np.asarray(consolidate_seg(seg_data))
    print(seg_data.shape,list(np.unique(seg_data)))
    seg_RGB=RGB_seg(seg_data)

    slice_RGB,seg_RGB = flip_img(slice_RGB,seg_RGB,f[0],f[1])

    fn=os.path.join(out_dir,'slice',os.path.basename(slice_fn).split('.nii.gz')[0]+'.rgb.input.{}{}.png'.format(f[0],f[1]))
    print(fn)
    # png.fromarray(slice_RGB.astype('uint8'),'L').save(fn)
    background=Image.fromarray(slice_RGB.astype('uint8'))
    background.save(fn)

    # slice_data=np.squeeze(rgb_2_lum(slice_RGB))
    # print(slice_data.shape,np.amax(slice_data),np.amin(slice_data))
    # fn=os.path.join(out_dir,os.path.basename(slice_fn).split('.nii.gz')[0]+'.lum.png')
    # print(fn)
    # Image.fromarray(slice_data.astype('uint8'), mode='L').save(fn)

    # seg_data=nib.load(segment_fn).get_data()
    # ch,num_ch_labeled=get_channel(seg_data)
    # if ch<0:
    #     print("{} does not have pink labels".format(segment_fn))
    # elif num_ch_labeled>1:
    #     print("{} has multiple channels with labels ch {} => 1".format(segment_fn,ch))
    #     ch=1
    # seg_data=np.squeeze(seg_data[:,:,ch])
    fn=os.path.join(out_dir,'segment',os.path.basename(segment_fn).split('.nii.gz')[0]+'.rgb.output.{}{}.png'.format(f[0],f[1]))
    print(fn)
    overlay=Image.fromarray(seg_RGB.astype('uint8'))
    overlay.save(fn)

    background = background.convert("RGBA")
    overlay = overlay.convert("RGBA")

    new_img = Image.blend(background, overlay, 0.2)
    fn=os.path.join(out_dir,'overlay',os.path.basename(slice_fn).split('.nii.gz')[0]+'_segmented.rgb.inandout.{}{}.png'.format(f[0],f[1]))
    print(fn)
    new_img.save(fn,"PNG") 

    
data_path='/data/shmuel/shmuel1/rap/histo/data/rm311_requad/'


# slices=['0122']
s='0122'
flip=[(1,1),(1,-1),(-1,1),(-1,-1)]
for f in flip:

    slices_fn = grab_files(data_path,'{}*slice.nii.gz'.format(s))
    segments_fn = grab_files(data_path,'{}*segmented.nii.gz'.format(s))
    files = split_train_valid(slices_fn,segments_fn)
    # slices = get_slices(files)
    files.sort()
    # print(files[:5])
    # print(slices)


    for (slice_fn,segment_fn) in files:
        print('{} : {}'.format(slice_fn,segment_fn))
        gen_png(slice_fn,segment_fn,f)




