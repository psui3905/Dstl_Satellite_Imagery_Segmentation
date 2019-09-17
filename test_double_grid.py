import os
from extra_functions import make_prediction, read_image, evaluate_accuracy, read_GT_Masks
import numpy as np
import pandas as pd
import cv2

import tensorflow as tf
from tf_unet.model import Model

import datetime

### PARAM LIST ###

# input data param ----------
mask 	 	=	-1		# Decide which class/classes to use. [0 - 9] for 1 class, -1 for 11 classes (5 for crop)
input_channels 	= 	3		# Number of input channels
output_classes 	= 	11		# Number of output classes
img_rows 	=	512		# Image patch size (row)
img_cols 	=	512	# Image patch size (column)
# ---------------------------

# u-net initialization params (later try to import this from pre-saved param file)
loss_function 	= 	"cross_entropy"	# cross_entropy, jaccard, DICE, cross_jac (cross_entropy+jaccard)
blocks 		=	7		# Number of "blocks"/"layer groups" for downsampling. That for upsampling is this number minus 1
layers 		=	2 		# Number of conv layers within one "block"/"layer group" 
cropping 	=	2*blocks*layers# crop the label
#cropping 	= 	35
features_root 	= 	32 		# Number of filters in starting layer
filter_size 	=	3
pool_size 	=	2
padding 	=	"SAME"		# Padding method: "SAME". "Valid" currently not support
regularizer	= 	"None"		# Regularisation on weights: "None", ("L2" not yet implemented)
activation 	=	"elu" 		# Activation function for conv layers "elu", "relu", "Leaky_relu"
batch_norm	= 	True		# batch normalisation
upsampling 	= 	1  		# 0: bilinear, 1: nearest neighbour, 2: bicubic, 3: area, 4: deconv
summaries 	=	True
# ---------------------------

# Testing Params
gpu_id 		= 	"0"		# GPU ID (Currently not implemented and assumed to be 0
#model_path 	= 	"./Pretrained_11_Classes"
#model_path 	= 	"./Pretrained_Crop_Class"
model_path	=	"./training_data_cross_7blk_b24_p256"
output_path	= 	"./Results"
test_ids 	=	['6110_3_1', '6110_3_1', '6110_3_1', '6110_3_1', '6110_3_1']
epsilon	= 	1e-12			# Small constant for calculate IoU
# ---------------------------

### END OF PARAM LIST ###

gs = pd.read_csv('../data/grid_sizes.csv', names=['ImageId', 'Xmax', 'Ymin'], skiprows=1)
allImageIds = gs.ImageId.unique()
#test_ids = allImageIds

COLORS = {
    1: [178, 178, 178],
    2: [102, 102, 102],
    3: [6, 88, 179],
    4: [125, 194, 223],
    5: [55, 120, 27],
    6: [160, 219, 166],
    7: [209, 173, 116],
    8: [180, 117, 69],
    9: [67, 109, 244],
    10:[39, 48, 215],
    11: [255, 255, 255]
}

def flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x

# create output directory if not exists
output_folder = os.path.join(output_path, os.path.basename(model_path))
if not os.path.exists(output_path):
    os.makedirs(output_path)
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# initialize model (u-net)
model = Model(channels=input_channels, 
    n_class=output_classes,
    img_rows=img_rows,
    img_cols=img_cols,
    cost=loss_function,
    cropping = cropping,
    batch_norm=batch_norm,
    cost_kwargs={"regularizer": regularizer},
    blocks=blocks,
    layers=layers,
    activation=activation,
    features_root=features_root,
    filter_size=filter_size,
    pool_size=pool_size,
    pad=padding,
    summaries=summaries)

def stretch_8bit(bands, lower_percent=2, higher_percent=98):
    out = np.zeros_like(bands).astype(np.float32)
    for i in range(3):
        a = 0
        b = 255
        c = np.percentile(bands[:, :, i], lower_percent)
        d = np.percentile(bands[:, :, i], higher_percent)
        t = a + (bands[:, :, i] - c) * (b - a) / (d - c)
        t[t < a] = a
        t[t > b] = b
        out[:, :, i] = t
    return out.astype(np.uint8)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = gpu_id
sess = tf.Session(config=config)

for image_id in test_ids:

    # Print image name
    print(image_id)

    # Read in the testing image
    image = read_image(image_id, bands=input_channels)

    c, h, w = image.shape
    h = int(int(h / img_rows) * img_rows)
    w = int(int(w / img_cols) * img_cols)

    # Make prediction ( Also for flip and swap_axis)
    predicted_mask_p1 = make_prediction(model, model_path, image, input_size=(img_rows, img_cols),
                                     crop=cropping, num_masks=output_classes, num_channels=input_channels, sess=sess)

    predicted_mask_p2 = make_prediction(model, model_path, image[:, int(img_rows/2):h-int(img_rows/2), int(img_cols/2):w-int(img_cols/2)], input_size=(img_rows, img_cols), crop=cropping, num_masks=output_classes, num_channels=input_channels, sess=sess)
  
    p2_mask = np.zeros((11, img_rows, img_cols))
    for i in range(img_rows):
        if i < img_rows / 2 - 1:
            j = int(img_cols / 2) - 2 - i
            p2_mask[:,i,0:j+1] = 1
            p2_mask[:,i,-j-1:] = 1
        elif i > img_rows / 2 + 1:  
            j = int(i - img_rows / 2 - 1)
            p2_mask[:,i,0:j+1] = 1
            p2_mask[:,i,-j-1:] = 1

    p1_mask = 1 - p2_mask

    for i in range(int(h / img_rows - 1)):
        for j in range(int( w / img_cols - 1)):
            predicted_mask_p1[:, img_rows * i + int(img_rows / 2): img_rows * (i+1) + int(img_rows / 2), img_cols * i + int(img_cols / 2): img_cols * (i + 1) + int(img_cols / 2)] *= p1_mask
            predicted_mask_p2[:, img_rows * i: img_rows * (i+1), img_cols * i: img_cols * (i + 1)] *= p2_mask

    predicted_mask_p1[:, int(img_rows / 2): int(h / img_rows - 1) * img_rows + int(img_rows / 2), int(img_cols / 2): int(h / img_cols - 1) * img_cols + int(img_cols / 2)] += predicted_mask_p2
    new_mask = predicted_mask_p1

    predicted_mask = read_GT_Masks('6110_3_1', 11, -1)
    predicted_mask = np.asarray(predicted_mask).astype(int) / 255

    # save to color image
    c, h, w = predicted_mask.shape
    for i in range(11):
        predicted_mask[i, :, :] = predicted_mask[i, :, :] * (i + 1)
    predicted_mask = np.sum(predicted_mask, axis=0, keepdims=False)

    color_image = np.zeros([h, w, 3])
    for i in range(h):
        for j in range(w):
            color_image[i, j, :] = COLORS[predicted_mask[i, j]]


    image = np.rollaxis(image, 0, 3)
    image = image[...,::-1]
    image = stretch_8bit(image)
    display_image = np.concatenate([color_image, image], axis=1)

    cv2.imwrite(os.path.join(output_folder, image_id + '.png'), display_image)

    #image_v = flip_axis(image, 1)
    #predicted_mask_v = make_prediction(model, model_path, image_v, input_size=(img_rows, img_cols),
    #                                   crop=cropping,
    #                                   num_masks=output_classes,
    #                                   num_channels=input_channels)

    #image_h = flip_axis(image, 2)
    #predicted_mask_h = make_prediction(model, model_path, image_h, input_size=(img_rows, img_cols),
    #                                   crop=cropping,
    #                                   num_masks=output_classes,
    #                                   num_channels=input_channels)

    #image_s = image.swapaxes(1, 2)
    #predicted_mask_s = make_prediction(model, model_path, image_s, input_size=(img_rows, img_cols),
    #                                   crop=cropping,
    #                                   num_masks=output_classes,
    #                                   num_channels=input_channels)

    # combine the predictions in different image angles
    #new_mask = np.power(predicted_mask * flip_axis(predicted_mask_v, 1) * flip_axis(predicted_mask_h, 2) * predicted_mask_s.swapaxes(1, 2), 0.25)


    # Evaluate Accuracy
    evaluate_accuracy(output_folder, new_mask, image_id, None, mask, epsilon)


