import numpy as np
import random
import h5py
import numpy as np
import cv2
import pandas as pd
import os
import extra_functions

def flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x

def form_batch(X, y, batch_size, num_channels, img_rows, img_cols, num_mask_channels):
    X_batch = np.zeros((batch_size, num_channels, img_rows, img_cols))
    y_batch = np.zeros((batch_size, num_mask_channels, img_rows, img_cols))
    X_height = X.shape[2]
    X_width = X.shape[3]

    for i in range(batch_size):
        random_width = random.randint(0, X_width - img_cols - 1)
        random_height = random.randint(0, X_height - img_rows - 1)

        random_image = random.randint(0, X.shape[0] - 1)

        y_batch[i] = y[random_image, :, random_height: random_height + img_rows, random_width: random_width + img_cols]
        X_batch[i] = np.array(X[random_image, :, random_height: random_height + img_rows, random_width: random_width + img_cols])

    return X_batch, y_batch

class Batch_generator_ram:

    def __init__(self, image_ids, img_rows, img_cols, num_channels, cropping, horizontal_flip=False, vertical_flip=False, swap_axis=False, mask=-1):
        self.CLASSES = {
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

        self.image_ids = image_ids
        train_wkt = pd.read_csv(os.path.join('../data', 'train_wkt_v4.csv'))
        shapes = pd.read_csv(os.path.join('../data', '3_shapes.csv'))
        train_shapes = shapes[shapes['image_id'].isin(train_wkt['ImageId'].unique())]
        min_train_height = train_shapes['height'].min()
        min_train_width = train_shapes['width'].min()
        self.y_train = np.zeros((24, 1, min_train_height, min_train_width))
        self.X_train = np.zeros((24, 3, min_train_height, min_train_width))

        for i in range(len(image_ids)):
            image_id = image_ids[i]
            # X_train
            image = extra_functions.read_image(image_id, 3)

            self.X_train[:] = image[:, :min_train_height, :min_train_width]

            # y_train
            _, height, width = image.shape
            self.y_train[:] = self.read_mask(image_id, height, width, mask)[:, :min_train_height, :min_train_width] / 255

            print("Loaded image {}".format(str(i)))

        self.y_train = np.reshape(self.y_train, [24, 1, min_train_height, min_train_width])


        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.swap_axis = swap_axis

        self.num_channels = num_channels
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.cropping = cropping

        if mask == -1:
            # set this value as variable later
            self.num_mask_channels = 11
        else:
            self.num_mask_channels = 1

    def __call__(self, batch_size):

        X_batch, y_batch = form_batch(self.X_train, self.y_train, batch_size, self.num_channels, self.img_rows, self.img_cols, self.num_mask_channels)

        for i in range(X_batch.shape[0]):
            xb = X_batch[i]
            yb = y_batch[i]
   
            if self.horizontal_flip:
                if np.random.random() < 0.5:
                    xb = flip_axis(xb, 1)
                    yb = flip_axis(yb, 1)

            if self.vertical_flip:
                if np.random.random() < 0.5:
                    xb = flip_axis(xb, 2)
                    yb = flip_axis(yb, 2)

            if self.swap_axis:
                if np.random.random() < 0.5:
                    xb = xb.swapaxes(1, 2)
                    yb = yb.swapaxes(1, 2)

            X_batch[i] = xb
            y_batch[i] = yb

        X_batch = np.transpose(X_batch, (0,2,3,1))
        y_batch = np.transpose(y_batch, (0,2,3,1))
        
        # crop the labels if necessary
        return X_batch, y_batch[:,self.cropping:self.img_rows-self.cropping,self.cropping:self.img_cols-self.cropping,:]

    def close(self):
        self.f.close()

    def read_mask(self, image_id, height, width, mask):

        if mask != -1:
            data = cv2.imread('../data/GT_Mask/' + image_id + '_' + self.CLASSES[mask+1] + '.png')
            data = np.transpose(data , (2,0,1))
            mask = data[0,:,:]
            return mask[None, :, :]
            

        mask = np.empty((0, height, width))

        for i in range(11):
            data = cv2.imread('../data/GT_Mask/' + image_id + '_' + self.CLASSES[i+1] + '.png')
            data = np.transpose(data , (2,0,1))
            data = data[0,:,:]
            mask = np.concatenate((mask, data[None, :,:]), axis=0)

        return mask
