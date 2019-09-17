import tensorflow as tf
import os
import tifffile as tiff

import cv2
import numpy as np

# Define classes
CLASS = {
        1: 'Building',
        2: 'Structure',
        3: 'Road',
        4: 'Track',
        5: 'Trees',
        6: 'Crops',
        7: 'Fast_H20',
        8: 'Slow_H20',
        9: 'Truck',
        10: 'Car',
        11: 'Background'}

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

# A band is not used ....
def read_image(image_id, bands):

    # read three band images
    img = np.transpose(tiff.imread("../data/three_band/{}.tif".format(image_id)), (1, 2, 0)) 
    img = stretch_8bit(img)
    #img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    #img[:,:,2] = img[:,:,2] * 1.5
    #img[img > 255] = 255
    #img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    #img = stretch_8bit(img, 1, 99)
    img = img / 255.0

    if bands == 16:
        img_m = np.transpose(tiff.imread("../data/sixteen_band/{}_M.tif".format(image_id)), (1, 2, 0)) / 2047.0
        img_p = tiff.imread("../data/sixteen_band/{}_P.tif".format(image_id)).astype(np.float32) / 2047.0

        height, width, _ = img_3.shape

        rescaled_M = cv2.resize(img_m, (width, height), interpolation=cv2.INTER_CUBIC)
        rescaled_P = cv2.resize(img_p, (width, height), interpolation=cv2.INTER_CUBIC)

        rescaled_M[rescaled_M > 1] = 1
        rescaled_M[rescaled_M < 0] = 0

        rescaled_P[rescaled_P > 1] = 1
        rescaled_P[rescaled_P < 0] = 0

        image_r = img_3[:, :, 0]
        image_g = img_3[:, :, 1]
        image_b = img_3[:, :, 2]
        nir = rescaled_M[:, :, 7]
        re = rescaled_M[:, :, 5]

        L = 1.0
        C1 = 6.0
        C2 = 7.5
        evi = (nir - image_r) / (nir + C1 * image_r - C2 * image_b + L)
        evi = np.expand_dims(evi, 2)

        ndwi = (image_g - nir) / (image_g + nir)
        ndwi = np.expand_dims(ndwi, 2)

        savi = (nir - image_r) / (image_r + nir)
        savi = np.expand_dims(savi, 2)

        ccci = (nir - re) / (nir + re) * (nir - image_r) / (nir + image_r)
        ccci = np.expand_dims(ccci, 2)

        rescaled_P = np.expand_dims(rescaled_P, 2)
    
        img = np.concatenate([rescaled_M, rescaled_P, ndwi, savi, evi, ccci, img_3], axis=2)

    result = np.transpose(img, (2, 0, 1)).astype(np.float16)
    return result


def make_prediction(model, model_path, X_train, input_size=(112, 112), crop=16, num_channels=3, num_masks=11, sess=None):

    out_h = input_size[0] - crop * 2
    out_w = input_size[1] - crop * 2

    height = X_train.shape[1]
    width = X_train.shape[2]

    num_h_tiles = int(np.ceil(height / out_h))
    num_w_tiles = int(np.ceil(width / out_w))

    rounded_height = num_h_tiles * out_h
    rounded_width = num_w_tiles * out_w

    padded_height = rounded_height + 2 * crop
    padded_width = rounded_width + 2 * crop

    padded = np.zeros((num_channels, padded_height, padded_width))

    padded[:, crop:crop + height, crop: crop + width] = X_train

    # add mirror reflections to the padded boundaries
    up = padded[:, crop:2 * crop, crop:-crop][:, ::-1]
    padded[:, :crop, crop:-crop] = up

    lag = padded.shape[1] - height - crop
    bottom = padded[:, height + crop - lag:crop + height, crop:-crop][:, ::-1]
    padded[:, height + crop:, crop:-crop] = bottom

    left = padded[:, :, crop:2 * crop][:, :, ::-1]
    padded[:, :, :crop] = left

    lag = padded.shape[2] - width - crop
    right = padded[:, :, width + crop - lag:crop + width][:, :, ::-1]
    padded[:, :, width + crop:] = right

    # steps for sliding window
    h_start = range(0, padded_height, out_h)[:-1]
    assert len(h_start) == num_h_tiles

    w_start = range(0, padded_width, out_w)[:-1]
    assert len(w_start) == num_w_tiles

    # getting sliding window patches
    print("[Prediction]: getting sliding window patches ...")
    temp = []
    for h in h_start:
        for w in w_start:
            temp += [padded[:, h:h + input_size[0], w:w + input_size[1]]]
    
    # make predictions
    print("[Prediction]: make predictions ...")
    prediction = model.predict(np.transpose(np.array(temp), (0,2,3,1)), model_path, sess)

    # translate argmax results into binary masks
    if num_masks == 11:
        tmp  = np.empty((prediction.shape[0], prediction.shape[1], prediction.shape[2], 0), dtype=np.int32)
        for i in range(11):
            pred = np.copy(prediction)
            pred[pred == i] = -1
            pred[pred != -1] = 0
            pred = pred * -1
            tmp = np.concatenate((tmp, pred), -1)
        prediction = tmp

    prediction = np.asarray(prediction)
    prediction = np.transpose(prediction, (0,3,1,2))

    # fit the prediction patches to the image mask
    print("[Prediction]: fit the prediction patches ...")
    predicted_mask = np.zeros((num_masks, rounded_height, rounded_width))

    for j_h, h in enumerate(h_start):
         for j_w, w in enumerate(w_start):
             i = len(w_start) * j_h + j_w
             predicted_mask[:, h: h + out_h, w: w + out_w] = prediction[i]

    return predicted_mask[:, :height, :width]


def read_GT_Masks(image_id, output_channels, mask):

    GT_masks = []

    if mask == -1:
        for cls in range(output_channels):
            GT_masks.append(cv2.imread('../data/GT_Mask/' + image_id + '_' + CLASS[cls+1] + '.png', 0))
    else:
        GT_masks.append(cv2.imread('../data/GT_Mask/' + image_id + '_' + CLASS[mask+1] + '.png', 0))

    return GT_masks

def evaluate_accuracy(output_folder, predicted_masks, image_id=None, GT_Masks=None, mask=None, epsilon=1e-12):

    # Read in the GT masks
    if GT_Masks == None:
         GT_Masks = read_GT_Masks(image_id, predicted_masks.shape[0], mask)

    Accum_IOU = 0
    # generate prediction for each class
    for i in range(predicted_masks.shape[0]):

        if predicted_masks.shape[0] > 1:
            mask = i
    
        # get GT_mask
        GT_Mask = GT_Masks[i]        

        # select and round mask to take values 0 or 1 and scale by 255
        Prediction = np.round(predicted_masks[i,:,:]) * 255

        # intersection & union
        intersection_mask = np.multiply(GT_Mask, Prediction)
        intersection = np.count_nonzero(intersection_mask)
        union = np.count_nonzero(GT_Mask) + np.count_nonzero(Prediction) - intersection
        IOU = intersection / (union + epsilon)

        # Y label are all zeros and predictions are all zeros => IoU is 1
        if np.count_nonzero(GT_Mask) == 0 and intersection == 0:
            IOU = 1.0
        Accum_IOU += IOU
        print("IOU for {: <10}:\t{:.4f}".format(CLASS[mask+1], IOU))
        
        # Save the prediction
        if output_folder != None:
            disp_img = np.concatenate((GT_Mask, Prediction), axis=1)
            disp_img = cv2.resize(disp_img, (1600,800))
            cv2.imwrite(os.path.join(output_folder, '{}_{:.4f}.png'.format(CLASS[mask+1], IOU)), disp_img)

    if predicted_masks.shape[0] > 1:
        print("Mean IOU          :\t{:.4f}".format(Accum_IOU / predicted_masks.shape[0]))


    return np.expand_dims(np.round(predicted_masks) * 255, axis=-1), Accum_IOU / predicted_masks.shape[0]
    
