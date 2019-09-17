"""
Script that caches train data for future training
"""

from __future__ import division

import os
import pandas as pd
import extra_functions
from tqdm import tqdm
import h5py
import numpy as np
import cv2

data_path = '../data'

train_wkt = pd.read_csv(os.path.join(data_path, 'train_wkt_v4.csv'))
gs = pd.read_csv(os.path.join(data_path, 'grid_sizes.csv'), names=['ImageId', 'Xmax', 'Ymin'], skiprows=1)

shapes = pd.read_csv(os.path.join(data_path, '3_shapes.csv'))

CLASSES = {
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

def read_mask(image_id, height, width):
    
    mask = np.empty((0, height, width))

    for i in range(11):
        data = cv2.imread('../data/GT_Mask/' + image_id + '_' + CLASSES[i+1] + '.png')
        data = np.transpose(data , (2,0,1))
        data = data[0,:,:]
        mask = np.concatenate((mask, data[None, :,:]), axis=0)

    return mask

def cache_train(bands):

    train_shapes = shapes[~shapes['image_id'].isin(train_wkt['ImageId'].unique())]

    print('num_train_images = ', train_shapes.shape[0])

    min_train_height = train_shapes['height'].min()
    min_train_width = train_shapes['width'].min()

    # num_train = train_shapes.shape[0]
    num_train = 50

    image_rows = min_train_height
    image_cols = min_train_width

    num_channels = bands

    num_mask_channels = 11

    f = h5py.File(os.path.join(data_path, 'unlabeled_train_3.h5'), 'w') 

    imgs = f.create_dataset('train', (num_train, num_channels, image_rows, image_cols), dtype=np.float16)
    # imgs_mask = f.create_dataset('train_mask', (num_train, num_mask_channels, image_rows, image_cols), dtype=np.uint8)

    ids = []

    i = 0
    for image_id in tqdm(sorted(train_shapes['image_id'].unique())):

        # drop the test image
        if image_id == '6110_3_1':
            continue

        image = extra_functions.read_image(image_id, bands)
        _, height, width = image.shape

        imgs[i] = image[:, :min_train_height, :min_train_width]

        # imgs_mask[i] = read_mask(image_id, height, width)[:, :min_train_height, :min_train_width] / 255

        ids += [image_id]
        i += 1

        if i == 50:
            break;

    # fix from there: https://github.com/h5py/h5py/issues/441
    f['train_ids'] = np.array(ids).astype('|S9')

    f.close()


if __name__ == '__main__':
    cache_train(bands=3)	# bands can be 3 or 16
