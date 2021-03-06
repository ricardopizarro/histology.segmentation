import nibabel as nib
import numpy as np
import glob
import json
import os
import sys
import difflib

from keras.models import model_from_json
from keras import backend as K
from keras.callbacks import ModelCheckpoint, History
from keras.optimizers import Adam

def grab_files(path,end):
    return glob.glob(path+end)

def get_coord(dim,tile_width):
    num_tiles = int(np.ceil(dim/tile_width))
    gap = (tile_width*num_tiles-dim)/(num_tiles-1)
    coord = [int(np.floor(i*(tile_width-gap))) for i in range(num_tiles)]
    return coord

def get_coord_random(dim,tile_width,nb_tiles):

    return list(np.random.random_integers(0,dim-tile_width,nb_tiles))

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


def normalize_tile(tile):
    m=float(np.mean(tile))
    st=float(np.std(tile))
    if st > 0:
        norm = (tile - m) / float(st)
    else:
        norm = tile - m
    return norm

def normalize_range(tile):
    if np.minimum(tile)<0:
        print('This tile has a minimum less than zero!')
    elif np.maximum(tile)>255:
        print('This tile has a maximum greater than 255!')
    return tile 


def normalize(tile_rgb):
    # normalize by RGB channel
    tile_norm=np.zeros(tile_rgb.shape)
    for ch in range(tile_rgb.shape[2]):
	tile=tile_rgb[:,:,ch]
	tile_norm[:,:,ch]=normalize_tile(tile)
	# tile_rgb[:,:,ch]=normalize_range(tile)
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

def gen_tiles_random(img_fn,segment_fn,files,nb_tiles=1,tile_width=2560):
    # img
    img = nib.load(img_fn)
    data = img.get_data()
    shape = img.shape

    # hull_data = nib.load(hull_fn).get_data()

    seg_data = nib.load(segment_fn).get_data()
    ch,num_ch_labeled=get_channel(seg_data)
    if ch<0:
        print("{} does not have pink labels".format(segment_fn))
    elif num_ch_labeled>1:
        print("{} has too many channels with multiple labels".format(segment_fn))
    # tile_width = 1600#560
    coord_x=get_coord_random(shape[0],tile_width,nb_tiles)
    coord_y=get_coord_random(shape[1],tile_width,nb_tiles)
    tiles = np.zeros([nb_tiles]+[tile_width]*2+[1])
    # print(tiles.shape)
    seg = np.zeros([nb_tiles]+[tile_width]*2)
    # print(seg.shape)
    tidx=0
    for tidx in range(nb_tiles):
        x = coord_x[tidx]
        y = coord_y[tidx]

        #seg_tile=seg_data[x:x+tile_width,y:y+tile_width,ch].tolist()
	#tile_tmp=data[x:x+tile_width,y:y+tile_width,:]
        #tile_tmp=segment(tile_tmp,seg_tile)
	#tile_tmp=normalize(tile_tmp)
	#tile_pad=np.zeros(tiles.shape[1:])
	#tile_pad[:tile_tmp.shape[0],:tile_tmp.shape[1]]=tile_tmp
        #tiles[tidx,:,:,:]=tile_pad
        data=rgb_2_lum(img.get_data())
        seg_tile=seg_data[x:x+tile_width,y:y+tile_width,ch].tolist()
        # hull_tile=hull_data[x:x+tile_width,y:y+tile_width,ch].tolist()
	tile_tmp=data[x:x+tile_width,y:y+tile_width,:]
	tile_norm=normalize(tile_tmp)
        tile_tmp=segment(tile_norm,seg_tile)
        tiles[tidx,:,:,:]=tile_tmp

        # channel ch determined above specifies Deepthy's labels
        seg_tmp=np.asarray(consolidate_seg(seg_tile))
        seg_pad=np.zeros(seg.shape[1:])
        try:
            seg_pad[:seg_tmp.shape[0],:seg_tmp.shape[1]]=seg_tmp
        except:
            print('Check the segmentation size: {}'.format(seg_tmp.shape))

        seg[tidx,:,:]=seg_pad

    return tiles,seg


def randomize(tiles,seg):
    nb_slices=tiles.shape[0]
    random_perm = np.random.permutation(nb_slices)
    tiles=tiles[random_perm]
    seg=seg[random_perm]
    return tiles,seg


def get_model(verbose=False):
    # dimension 1600x1600
    fn = "../model/NN_brown_unet_d2560_c5p2.model.json"
    with open(fn) as json_data:
        d = json.load(json_data)
    model = model_from_json(d)
    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])
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


def save_history(path,performance):
    # track performance (accuracy and loss) for training and validation sets
    # after each epoch completes and save as .json string
    json_string=json.dumps(performance.history)
    fn=os.path.join(path,'history_performance.json')
    with open(fn, 'w') as outfile:
        json.dump(json_string, outfile)

def runNN(train_files,valid_files):
    # nb_step is number of tiles per step
    input_size=(2560,2560,1)
    output_size=(2560,2560,1)
    # number of tiles per step
    nb_step=1 #20
    model = get_model(verbose=True)
    # track performance (dice coefficient loss) on train and validation datasets
    performance = History()
    checkpath="../weights/weights.brown.d2560/nohull_001/weights.{epoch:03d}_of_100_dice_coef_loss_{loss:0.2f}.h5"
    checkpointer=ModelCheckpoint(checkpath, monitor='loss', verbose=0, save_best_only=True, save_weights_only=True, mode='min')
    # fit the model using the data generator defined below
    # model.fit_generator(fileGenerator(train_files,nb_step=nb_step,verbose=False,input_size=input_size,output_size=output_size), steps_per_epoch=400, epochs=1000, verbose=1,
    model.fit_generator(fileGenerator(train_files,nb_step=nb_step,verbose=False,input_size=input_size,output_size=output_size), steps_per_epoch=40, epochs=100, verbose=1,
            validation_data=fileGenerator(valid_files,nb_step=1,verbose=False,input_size=input_size,output_size=output_size),validation_steps=1,callbacks=[performance,checkpointer])
    # save the weights at the end of epochs
    model.save_weights('../weights/weights.brown.d2560/nohull_001/weights.100ep.FINAL.h5',overwrite=True)
    # save the performance (accuracy and loss) history
    save_history(os.path.dirname(checkpath),performance)

def fileGenerator(files,nb_step=1,verbose=True,input_size=(2560,2560,1),output_size=(2560,2560,1)):
    X = np.zeros((nb_step,) + input_size )
    Y = np.zeros((nb_step,) + output_size )
    n = 0
    while True:
        while n < nb_step:
            try:
                i = np.random.randint(0,len(files))
		# print(i)
                slice_fn=files[i][0]
                segment_fn=files[i][1]
                # slice_nb = os.path.basename(segment_fn)[:4]
                # hull_dir='/data/shmuel/shmuel1/rap/histo/data/hull/'
                # hull_list=glob.glob(hull_dir+'*.gz')
                # hull_bn=[os.path.basename(fn) for fn in hull_list]
                # hull_fn=difflib.get_close_matches(os.path.basename(segment_fn),hull_bn)[0]
                # if not hull_fn:
                #     print('We aint found no hull for {}'.format(segment_fn))
                # hull_fn=hull_dir+hull_fn
                if verbose:
                    print("{} : {}".format(slice_fn,segment_fn))
                # tiles,seg=gen_tiles(slice_fn,segment_fn)
                tile_width=input_size[0]
                tiles,seg=gen_tiles_random(slice_fn,segment_fn,files,nb_step,tile_width)
                # tiles,seg=randomize(tiles,seg)
                nb_slices=tiles.shape[0]
                seg=seg.reshape((nb_slices,)+output_size)
                if nb_step-n < nb_slices:
                    need_slices=nb_step-n
                    X[n:n+need_slices]=tiles[:need_slices]
                    Y[n:n+need_slices]=seg[:need_slices]
                else:
                    X[n:n+nb_slices]=tiles
                    Y[n:n+nb_slices]=seg
            	    # print(X.shape)
                n+=nb_slices
            except Exception as e:
                print(str(e))
                pass
	if X.size:
            yield X,Y
	else:
	    print("X is empty!!!")
	    continue


def split_train_valid(slices_fn,segments_fn):
    # print(sorted(slices_fn))
    train_files =[] 
    validation_files=[]
    for n,segment_fn in enumerate(segments_fn):
        # slice_quad_number = os.path.basename(segment_fn).split('segmented.nii')[0].upper()
        # print(os.path.basename(segment_fn))
        slice_fn=difflib.get_close_matches(segment_fn,slices_fn)[0]
        # slice_fn=[fn for fn in slices_fn if slice_quad_number in fn.upper()]
        if not slice_fn:
            print("Could not find an equivalent segment file {}".format(segment_fn))
            continue
        # print(slice_fn)
        if np.mod(n,5):
            train_files.append((slice_fn,segment_fn))
        else:
            validation_files.append((slice_fn,segment_fn))
        # print(slice_number)
    # print(sorted(segments_fn))
    return train_files,validation_files


def split_cross_valid(slices_fn,segments_fn,train,valid,test):
    train_files =[] 
    validation_files=[]
    test_files=[]
    for n,segment_fn in enumerate(segments_fn):
        slice_fn=difflib.get_close_matches(segment_fn,slices_fn)[0]
        if not slice_fn:
            print("Could not find an equivalent segment file {}".format(segment_fn))
            continue
        slice_nb = os.path.basename(segment_fn)[:4]
        if slice_nb in train:
            train_files.append((slice_fn,segment_fn))
        elif slice_nb in valid:
            validation_files.append((slice_fn,segment_fn))
        elif slice_nb in test:
            test_files.append((slice_fn,segment_fn))
        else:
            print('{} is not in any subset!'.format(segment_fn))
    return train_files,validation_files,test_files
            


# data_path = "/data/shmuel/shmuel1/rap/histo/data/"
data_path = "/data/shmuel/shmuel1/deepthi/RM311_HighRes_Seg_Set1_1-74/"
if not os.access(data_path, os.R_OK):
    print('Cannot read any of the files in {}'.format(data_path))
    sys.exit()

slices=glob.glob(data_path+"*")
slices=[os.path.basename(s) for s in slices]
# print(slices)

test=[s for i,s in enumerate(slices) if np.mod(i,5)==0]
valid=[s for i,s in enumerate(slices) if np.mod(i,5)==1]
train=[s for i,s in enumerate(slices) if np.mod(i,5)>1]

slices_fn = grab_files(data_path,"*/*.jpg.nii")
segments_fn = grab_files(data_path,"*/*segmented.nii*")

train_files,validation_files,test_files = split_cross_valid(slices_fn,segments_fn,train,valid,test)

# files=train_files+validation_files


'''
all_files=train_files+validation_files
for line in [all_files[0]]:
    slice_fn=line[0]
    segment_fn=line[1]
    tiles,seg=gen_tiles_random(slice_fn,segment_fn,nb_tiles=1)
'''

# print(files)

print('\n==Training NN UNET ==\n')
runNN(train_files,validation_files)


