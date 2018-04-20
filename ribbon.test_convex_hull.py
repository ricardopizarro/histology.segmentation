import mahotas
# Citation:Coelho, L.P. 2013. Mahotas: Open source software for scriptable computer vision. Journal of Open Research Software 1(1):e3, DOI: http://dx.doi.org/10.5334/jors.ac
import nibabel as nib
import numpy as np
import cv2 as cv
import glob
import os
from scipy import ndimage
import difflib
from subprocess import check_call

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def grab_files(path,end):
    return glob.glob(path+end)


def consolidate(seg):
    # swap elements labeled 6 and 2 to 0.  
    # elements labeled 6 indicate a tear in the white matter
    # elements labeled 5 indicate a fold in the gray matter
    # elements labeled 4 indicate subcortex or cerebellum
    # elements labeled 2 indicate a tear in the gray matter
    # elements labeled 3 indicate a blood vessel in the gray matter
    for m,row in enumerate(seg):
        for n,elem in enumerate(row):
            if elem in range(2,7):
                seg[m][n]=1
    return seg


def segment(tile,seg):
    # We used the labeled seg to segment the subcortex and cerebellum
    # To mask this portion out we simply make it white like the background [255,255,255]
    for m,row in enumerate(seg):
        for n,elem in enumerate(row):
            if elem in [4]:
                tile[m,n,:]=10
    return tile

def swap_labels(tile,a,b):
    # we wish to swap elements in order to change the color in itksnap
    # this function swaps a for b
    for m,row in enumerate(tile):
        for n,elem in enumerate(row):
            if elem in [a]:
                tile[m][n]=b
    return tile

def normalize_tile(tile):
    m=float(np.mean(tile))
    st=float(np.std(tile))
    if st > 0:
        norm = (tile-m)/float(st)
    else:
        norm = tile - m
    return norm

def normalize(tile_rgb):
    # normalize by RGB channel
    tile_norm=np.zeros(tile_rgb.shape)
    for ch in range(tile_rgb.shape[2]):
	tile=tile_rgb[:,:,ch]
        tile_norm[:,:,ch]=normalize_tile(tile_rgb[:,:,ch])
    return tile_norm

def get_channel(img):
    ch_ret=-1
    num_ch_labeled=0
    for ch in range(img.shape[2]):
        # print(ch)
        # print(np.unique(img[:,:,ch]))
        if len(np.unique(img[:,:,ch]))>2:
            ch_ret=ch
            num_ch_labeled+=1
    return ch_ret,num_ch_labeled



def get_edges(img):
    # print(img.shape[1:3])
    img=np.reshape(img,img.shape[1:3])
    sx = ndimage.sobel(img, axis=0, mode='constant')
    sy = ndimage.sobel(img, axis=1, mode='constant')
    sob = np.around(np.hypot(sx, sy))
    mag = np.max(sob)-np.min(sob)
    if mag > 1e-3:
        sob = np.around((sob-np.min(sob))/(mag))
    sob_dilated = ndimage.binary_dilation(sob).astype(sob.dtype)
    sob_dilated = ndimage.binary_dilation(sob_dilated).astype(sob.dtype)
    return sob,sob_dilated



def save_to_nii(data,fn):
    out_dir='/data/shmuel/shmuel1/rap/histo/data/20180420_hull/'
    affine=np.eye(len(data.shape))
    img = nib.Nifti1Image(data,affine)
    path=out_dir+fn
    print(path)
    nib.save(img,path)
    check_call(['gzip',path])

def bgr_2_gray(img):
    # the bgr channel is located at axis=2
    img=0.2989*img[:,:,2]+0.587*img[:,:,1]+0.114*img[:,:,0]
    return img

def rgb_2_gray(img):
    # the rgb channel is located at axis=3 for the tiles
    img=0.2989*img[:,:,:,0]+0.587*img[:,:,:,1]+0.114*img[:,:,:,2]
    return img



def convex_hull_segment(fdata):
    # Takes the range of values in the volume and creates a subvolume and calls a convexhull fill method
    # This uses mahatos based fill_convexhull method 
    # First two x slices are returning garbage values so removing those slices from the volume.

    # finput_file=plt.imread(finput_volume)
    # finput_file = rgb2gray(finput_file) 
    # fdata=finput_file

    zeros=np.zeros((fdata.shape[0],fdata.shape[1]))
    points=np.array(np.where(fdata[:,:]>0.0))
    if len(points)>=0:

	returnedCanvas=mahotas.polygon.fill_convexhull((fdata[:,:]>0.0)*1)
	zeros[2:-2,2:-2]=returnedCanvas[2:-2,2:-2]
    	# plt.imshow(zeros)
	# plt.show()
    else:
	print("No points in this slice to be hulled.")

    # Saving data as convex_hull.nii.gz
    #img = nib.Nifti1Image(zeros, finput_file.affine) # Creating array of zeros with the shape of final_brain_volume_mask
    #img.to_filename(os.path.join(fprefix,outputfile))
    return zeros



def convexicate(slice_fn,segment_fn,verbose=False):

    slice_fn=line[0]
    segment_fn=line[1]
    slice_nb=os.path.basename(segment_fn)[:4]
    quad_nb=slice_fn.split('.jpg.')[0][-1]
    if verbose:
        print("{} : {}".format(slice_fn,segment_fn))
    segment_data = nib.load(segment_fn).get_data()
    ch,num_ch_labeled=get_channel(segment_data)
    if ch<0:
        print("{} does not have pink labels".format(segment_fn))
    elif num_ch_labeled>1:
        print("{} has too many channels with multiple labels".format(segment_fn))
    segment_data=segment_data[:,:,ch]
    print(segment_data.shape)
    segment_data = np.asarray(consolidate(segment_data.tolist()))
    segment_hull = convex_hull_segment(segment_data)
    segment_hull = np.reshape(segment_hull,segment_hull.shape+(1,1,))
    segment_hull = np.repeat(segment_hull,3,axis=2)
    fn='{0}_6x_concat_6x_whole.jpg_{1}.hull.nii'.format(slice_nb,quad_nb)
    save_to_nii(segment_hull,fn)





slice_fn = sys.argv[1]
segment_fn = sys.argv[2]

print('\n==Convexicating the file ==\n')
convexicate(slice_fn,segment_fn,verbose=True)

# filename=sys.argv[1]
# ConvexHullCerebellum(fprefix=os.getcwd(),finput_volume=filename,outputfile='test_hull.png')

