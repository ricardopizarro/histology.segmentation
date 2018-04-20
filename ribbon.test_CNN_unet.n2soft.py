import nibabel as nib
import numpy as np
import cv2 as cv
import glob
import json
import os
import tensorflow as tf
from scipy import ndimage
import difflib
import random
from subprocess import check_call

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from keras.models import model_from_json
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.utils import np_utils

def grab_files(path,end):
    return glob.glob(os.path.join(path,end))

def get_coord_random(dim,tile_width,nb_tiles):
    # nx is the number of tiles in the x-direction to cover the edge
    nx=int(np.ceil(float(dim[0])/tile_width))
    # ny is the number of tiles in the y-direction to cover the edge
    ny=int(np.ceil(float(dim[1])/tile_width))

    gap = (tile_width*nx-dim[0])/(nx-1)
    # uniformly sample along one dimension to cover the edge
    uni_x = [int(np.floor(i*(tile_width-gap))) for i in range(nx)]
    uni_x[-1]=dim[0]-tile_width
    edge_x=[0]*ny+[dim[0]-tile_width]*ny+uni_x*2
    x=list(np.random.random_integers(0,dim[0]-tile_width,nb_tiles))
    x=edge_x+x

    gap = (tile_width*ny-dim[1])/(ny-1)
    # uniformly sample along one dimension to cover the edge
    uni_y = [int(np.floor(i*(tile_width-gap))) for i in range(ny)]
    uni_y[-1]=dim[1]-tile_width
    edge_y=uni_y*2+[0]*nx+[dim[1]-tile_width]*nx
    y=list(np.random.random_integers(0,dim[1]-tile_width,nb_tiles))
    y=edge_y+y

    # nb_tiles = 2*int(np.ceil(float(dim)/tile_width))
    # gap = (tile_width*nb_tiles-dim)/(nb_tiles-1)
    # coord = [int(np.floor(i*(tile_width-gap))) for i in range(nb_tiles)]
    # coord[-1]=dim-tile_width
    return zip(x,y)


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
                seg[m][n]=1
            elif elem in [2,4,6]:
                seg[m][n]=0
    return seg


def segment(tile,seg,hull):
    # We used the labeled seg to segment the subcortex and cerebellum
    # To mask this portion out we simply make it white like the background post normalization [10,10,10]
    for m,row in enumerate(seg):
        for n,elem in enumerate(row):
            # print('we aint got no hull')
            # print(hull[m][n])
            if elem in [4]:
                tile[m,n]=10
            # elif 0 in hull[m][n]:
            #     tile[m,n,:]=20
    return tile

def convexicate(data,hull):
    # We used the labeled hull to segment the brain from background
    # To mask this portion out we simply make it white like the background [255,255,255]
    for m,row in enumerate(hull):
        for n,elem in enumerate(row):
            if elem in [0]:
                data[m,n,:]=255
    return data


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

def rgb_2_lum(img):
    # the rgb channel is located at axis=2 for the data
    img=0.2126*img[:,:,0]+0.7152*img[:,:,1]+0.0722*img[:,:,2]
    return img

def gen_tiles(img_fn,seg_fn,hull_fn,tile_width,slice_nb,quad_nb,nb_tiles):
    img = nib.load(img_fn)
    data = rgb_2_lum(img.get_data())
    shape = img.shape

    if hull_fn:
        hull_data = nib.load(hull_fn).get_data()
    else:
        hull_data=np.zeros(shape)+1

    seg_data = nib.load(seg_fn).get_data()
    ch,num_ch_labeled=get_channel(seg_data)
    if ch<0:
        print("{} does not have pink labels".format(seg_fn))
    elif num_ch_labeled>1:
        print("{} has too many channels with multiple labels".format(seg_fn))
    # tile width will allow for three layers of 2x2 max pooling, divisible by 8=2*2*2
    coord=get_coord_random(shape,tile_width,nb_tiles)
    coord=sorted(list(set(coord)))
    print(coord)
    nb_tiles=len(coord)
    fn='{0}_6x_concat_6x_whole.jpg_{1}.orig.jpg.nii'.format(slice_nb,quad_nb,nb_tiles)
    save_to_nii(np.reshape(data,shape+(1,1,)),fn)

    # tiles should have dimension (20-30,560,560,3)
    tiles = np.zeros([nb_tiles]+[tile_width]*2+[1])
    tiles_lum = np.zeros([nb_tiles]+[tile_width]*2+[1])
    seg = np.zeros([nb_tiles]+[tile_width]*2)
    tidx=0
    for x,y in coord:
        print((tidx,x,y))
        # data=rgb_2_lum(img.get_data())
        seg_tile=seg_data[x:x+tile_width,y:y+tile_width,ch].tolist()
        hull_tile=hull_data[x:x+tile_width,y:y+tile_width,ch].tolist()
        tile_lum=data[x:x+tile_width,y:y+tile_width]
        tile=normalize_tile(tile_lum)
        # tile=dropout(tile,drop)
        tile=segment(tile,seg_tile,hull_tile)
        seg_tmp=np.asarray(consolidate_seg(seg_tile))
        seg_pad=np.zeros(seg.shape[1:])
        try:
            seg_pad[:seg_tmp.shape[0],:seg_tmp.shape[1]]=seg_tmp
        except:
            print('Check the segmentation size: {}'.format(seg_tmp.shape))

        tiles[tidx,:,:,0]=tile
        tiles_lum[tidx,:,:,0]=tile_lum
        seg[tidx,:,:]=seg_pad
        tidx+=1
    return tiles_lum,tiles,seg,coord,shape


def avg_tile(slice_avg,single_tile,x,y,tile_width):
    slice_sum=slice_avg[0]
    slice_sum[x:x+tile_width,y:y+tile_width]+=single_tile

    slice_count=slice_avg[1]
    slice_count[x:x+tile_width,y:y+tile_width]+=1
    return slice_sum,slice_count


def retile(tiles,coord,slice_shape,tile_width):
    # slice_shape is rgb shape with a 3 at the end
    nb_tiles=tiles.shape[2]
    # typical size: (25,2666,2760)
    slice_sum=np.zeros(slice_shape[:-1])
    slice_count=np.zeros(slice_shape[:-1])
    slice_avg=[slice_sum,slice_count]
    # tabulate the elements here, we will do a final mode at the end
    slice = np.zeros(slice_shape[:-1])
    tidx=0
    for x,y in coord:
        single_tile=tiles[:,:,tidx,0]
        slice_avg=avg_tile(slice_avg,single_tile,x,y,tile_width)
        tidx+=1
    # print(np.unique(slice_count))
    slice=np.true_divide(slice_avg[0],slice_avg[1])
    # flip slice to make equivalent to original dataset
    # slice=slice[::-1,::-1]
    slice=np.reshape(slice,slice.shape+(1,1,))
    return slice


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


def get_model(verbose=False):
    # fn = "../model/NN_brown_unet.model.json"
    fn = "../model/NN_brown_unet_d2560_c5p2.n2soft.model.json"
    with open(fn) as json_data:
        d = json.load(json_data)
    model = model_from_json(d)
    model.compile(optimizer=Adam(lr=1e-5), loss='categorical_crossentropy', metrics=[dice_coef])
    if verbose:
        print(model.summary())
    return model

def dice_coef(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def calc_dc(data):
    data=data.astype(np.int64)
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

def save_to_nii(data,fn):
    out_dir='/data/shmuel/shmuel1/rap/histo/prediction/subcortex/20180330_nohull/'
    affine=np.eye(len(data.shape))
    img = nib.Nifti1Image(data,affine)
    path=os.path.join(out_dir,fn)
    print(path)
    nib.save(img,path)
    check_call(['gzip', path])

def bgr_2_gray(img):
    # the bgr channel is located at axis=2
    img=0.2989*img[:,:,2]+0.587*img[:,:,1]+0.114*img[:,:,0]
    return img

def rgb_2_gray(img):
    # the rgb channel is located at axis=3 for the tiles
    img=0.2989*img[:,:,:,0]+0.587*img[:,:,:,1]+0.114*img[:,:,:,2]
    return img

def annotate(tmp,dc):
    img=tmp
    # color is BGR, not RGB
    img_shape=img.shape
    img=np.swapaxes(img,0,1)
    img=img[::-1,::-1]
    print(img.shape)
    img=np.reshape(img,img.shape+(1,))
    img=np.repeat(img,[3],axis=2)
    print(img.shape)
    # img = cv.rectangle(img,(450,450),(500,500),(0,100,0),3)
    cv.putText(img,'{0:0.2f}'.format(dc),(100,100),cv.FONT_HERSHEY_PLAIN,3,(125,125,125),3,cv.LINE_AA)
    img=bgr_2_gray(img)
    img=np.reshape(img,img_shape)
    img=img[::-1,::-1]
    img=np.swapaxes(img,0,1)
    img=np.maximum(img,tmp)
    return img

def gen10tiles(slice_fn,tile_width):
    tmp=nib.load(slice_fn).get_data()
    # img=tmp
    # color is BGR, not RGB
    img_shape=tmp.shape
    img = np.zeros(img_shape,np.uint8)
    # m = tmp.transpose((1, 2, 0)).astype(np.uint8).copy() 
    # img=np.swapaxes(img,0,1)
    # img=img[::-1,::-1]
    # print(img.shape)
    # img.astype('float64')
    # print(img.dtype)
    nb_tiles=17
    x=list(np.random.random_integers(0,img_shape[1]-tile_width,nb_tiles))
    y=list(np.random.random_integers(0,img_shape[0]-tile_width,nb_tiles))
    for n in range(nb_tiles):
        top_left=(x[n],y[n])
        print(top_left)
        bottom_right=(x[n]+tile_width,y[n]+tile_width)
        print(bottom_right)
        cv.rectangle(img,top_left,bottom_right,(255,255,255),10)
    # img = m.transpose((2, 0, 1)).astype(tmp.dtype).copy()
    # img=np.maximum(img,tmp)
    print(img.shape)
    img=np.around(bgr_2_gray(img))
    print(img.shape)
    img=np.reshape(img,img_shape[:-1]+(1,1,))
    # img=img[::-1,::-1]
    # img=np.swapaxes(img,0,1)
    return img


def testNN(files,nb_tiles_in,verbose=False):
    # nb_step is number of tiles per step
    input_size=(2560,2560,1)
    output_size=(2560,2560,2)
    batch_size=32
    tile_width=2560

    model = get_model(verbose=True)
    # weights_fn='../weights/weights.brown.d2560/nohull_003/weights.set027.epochs2800.FINAL.h5'
    weights_fn='../weights/weights.brown.d2560/nohull_003/weights.set050.epochs5400.FINAL.h5'
    model.load_weights(weights_fn)

    for l,line in enumerate(files):
        try:
            slice_fn=line[0]
            segment_fn=line[1]
            slice_nb=os.path.basename(segment_fn)[:4]
            hull_fn=[]#os.path.join(os.path.dirname(slice_fn),'hull','{}-segment_hull.nii.gz'.format(slice_nb))
            if verbose:
                print("{} : {} : {}".format(slice_fn,segment_fn,hull_fn))
            # tiles_rgb are for viewing, tiles are normalized used for predicting
            # seg is segmentation done by deepthy
            # coord identifies the location of the tile,seg
            # slice_shape specifies the shape of the image
            nb_tiles=nb_tiles_in
            tiles_lum,tiles,seg,coord,slice_shape=gen_tiles(slice_fn,segment_fn,hull_fn,tile_width,nb_tiles)
            print('did we gen tiles?')
            nb_tiles=len(coord)
            print(seg.shape)
            seg=np.reshape(np_utils.to_categorical(seg,output_size[-1]),(nb_tiles,)+output_size)
            output_size_tiles=output_size[:-1]+(nb_tiles,)+(2,)
            input_size_tiles=output_size[:-1]+(nb_tiles,)+(1,)
            print((input_size_tiles,output_size_tiles))
            X_test_tiles=np.reshape(tiles_lum,input_size_tiles)
            y_out_tiles=np.zeros(output_size_tiles)
            y_true_tiles=np.zeros(output_size_tiles)
            y_pred_tiles=np.zeros(output_size_tiles)
            dc_val=np.zeros(nb_tiles)

            X_test_slice=retile(X_test_tiles,coord,slice_shape,tile_width)
            fn='{0}-sliceindex_{1:04d}tiled.nii.gz'.format(slice_nb,nb_tiles)
            save_to_nii(X_test_slice,fn)

            for n in range(nb_tiles):
                X_test=np.reshape(tiles[n],(1,)+input_size)
                y_true=np.reshape(seg[n],(1,)+output_size)
                # y_edge_thin,y_edge_thick=get_edges(y_true)
                y_pred = np.around(model.predict(X_test,batch_size=batch_size,verbose=0))
                # thresh = 0.5
                y_pred=y_pred.astype(dtype=y_true.dtype)
                # getting the dice coefficient
                data12 = y_true[0]+2*y_pred[0]
                print(data12.shape)
                dc_man = calc_dc(data12.flatten())
                print('This is dice: {0:0.4f}'.format(dc_man))
                dc_val[n]=float(dc_man)

                img = y_true[0,:,:,:]+3*y_pred[0,:,:,:]
                y_true_tiles[:,:,n,:]=y_true[0]
                y_pred_tiles[:,:,n,:]=y_pred[0]
                y_out_tiles[:,:,n,:] =img

            print('{} : {}'.format(slice_nb,nb_tiles))
            fn='{0}-true_segment_{1:04d}tiles.nii.gz'.format(slice_nb,nb_tiles)
            # save_to_nii(y_true_tiles,fn)
            fn='{0}-segmented_{1:04d}tiles.nii.gz'.format(slice_nb,nb_tiles)
            # save_to_nii(y_out_tiles,fn)
            fn='{0}-sliceindex_{1:04d}tiles.nii.gz'.format(slice_nb,nb_tiles)
            # save_to_nii(X_test_tiles,fn)

            Y_true_slice=retile(y_true_tiles,coord,slice_shape,tile_width)
            Y_true_slice=np.around(Y_true_slice)
            print(np.unique(Y_true_slice))
            fn='{0}-true_segment_{1:04d}tiled.nii.gz'.format(slice_nb,nb_tiles)
            save_to_nii(Y_true_slice,fn)

            Y_pred_slice=retile(y_pred_tiles,coord,slice_shape,tile_width)
            Y_pred_slice=np.around(Y_pred_slice)
            print(np.unique(Y_pred_slice))
            fn='{0}-pred_segment_{1:04d}tiled_dc{2:0.3f}.nii.gz'.format(slice_nb,nb_tiles,np.mean(dc_val))
            save_to_nii(Y_pred_slice,fn)
           
            data12_slice = Y_true_slice+2*Y_pred_slice
            dc_slice = calc_dc(data12_slice.flatten())
            print('Dice coefficient ( avg tiles | slice ) : ( {0:0.3f} | {0:0.3f} )'.format(np.mean(dc_val),dc_slice))

            bins = np.linspace(0, 1, 11)
            plt.hist(dc_val,bins)
            plt.title("Dice Coefficient Distribution")
            plt.xlabel("Dice Coefficient")
            plt.ylabel("Frequency")
            fn='{0}-dice_distr_{1:04d}tiled_dc_avg{2:0.3f}.png'.format(slice_nb,nb_tiles,np.mean(dc_val))
            plt.savefig('/data/shmuel/shmuel1/rap/histo/prediction/subcortex/20180330_nohull/'+fn)
            plt.close()

            # Y_out_slice=retile(y_out_tiles,coord,slice_shape,tile_width)
            Y_out_slice=Y_true_slice+3*Y_pred_slice
            fn='{0}-segmented_{1:04d}tiled_dc{2:0.3f}.nii.gz'.format(slice_nb,nb_tiles,np.mean(dc_val))
            # save_to_nii(Y_out_slice,fn)

        except Exception as e:
            print(str(e))
            pass

 
def split_train_valid(slices_fn,segments_fn):
    # print(sorted(slices_fn))
    train_files =[] 
    validation_files=[]
    for n,seg_fn in enumerate(segments_fn):
        slice_number = os.path.basename(seg_fn)[:4]
        slice_fn=[fn for fn in slices_fn if slice_number in fn]
        if not slice_fn:
            print("Could not find an equivalent slice for number {}".format(slice_number))
            continue
        # print(slice_fn)
        if np.mod(n,5):
            train_files.append((slice_fn[0],seg_fn))
        else:
            validation_files.append((slice_fn[0],seg_fn))
        # print(slice_number)
    # print(sorted(segments_fn))
    return train_files,validation_files


def split_cross_valid(slices_fn,segments_fn,train,valid):
    # slices_base=[os.path.basename(s) for s in slices_fn]
    # segments_base=[os.path.basename(s) for s in segments_fn]

    train_files =[]
    validation_files=[]
    for n,seg_fn in enumerate(segments_fn):
        slice_fn=difflib.get_close_matches(seg_fn,slices_fn)[0]
        # slice_fn=[s for s in slices_fn if slice_base in s]
        if not slice_fn:
            print("Could not find an equivalent segment file {}".format(segment_fn))
            continue
        slice_nb = os.path.basename(seg_fn)[:4]
        if slice_nb in train:
            train_files.append((slice_fn,seg_fn))
        elif slice_nb in valid:
            validation_files.append((slice_fn,seg_fn))
        else:
            print('{} is not in any subset!'.format(segment_fn))

    return train_files,validation_files



slices_fn=[]
segments_fn=[]
slices=[]
# data_path = "/data/shmuel/shmuel1/mok/histology_nhp/segmentation/HBP2/"
data_path = ['/data/shmuel/shmuel1/deepthi/RM311_HighRes_Seg_Set1_1-70','/data/shmuel/shmuel1/deepthi/RM311_HighRes_Seg_Set2_71-113']
for p in data_path:
    ss=grab_files(p,'*')
    slices+=[os.path.basename(s) for s in ss]
    slices_fn += grab_files(p,'*/*jpg.nii')
    segments_fn += grab_files(p,'*/*segmented.nii*')

# print(segments_fn)

valid=[s for i,s in enumerate(slices) if np.mod(i,5)==0]
train=[s for i,s in enumerate(slices) if np.mod(i,5)>0]

train_files,validation_files = split_cross_valid(slices_fn,segments_fn,train,valid)

# print(sorted(train_files))

print('\n==Testing NN UNET ==\n')
nb_tiles=50
# slices: 725,505,765,750,690
# index: 5,0,4,2,10
# runNN(all_train_files,all_valid_files,NN,nb_mods,nb_step,input_size)
# validation_files=train_files+validation_files
# print(validation_files)
# valid_len=len(validation_files)
# validation_files=[validation_files[i] for i in range(13,valid_len)]
# testNN([validation_files[0]],nb_tiles,verbose=True)
testNN([validation_files[0]],nb_tiles,verbose=True)
