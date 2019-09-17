import numpy as np
import random
import h5py

def flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x

def form_batch(X, y, batch_size, num_channels, img_rows, img_cols, num_mask_channels, unlabeled=False):
    
    X_batch = np.zeros((batch_size, num_channels, img_rows, img_cols))

    if not unlabeled:
        y_batch = np.zeros((batch_size, num_mask_channels, img_rows, img_cols))

    X_height = X.shape[2]
    X_width = X.shape[3]

    for i in range(batch_size):
        random_width = random.randint(0, X_width - img_cols - 1)
        random_height = random.randint(0, X_height - img_rows - 1)

        random_image = random.randint(0, X.shape[0] - 1)

        if not unlabeled:
            y_batch[i] = y[random_image, :, random_height: random_height + img_rows, random_width: random_width + img_cols]

        X_batch[i] = np.array(X[random_image, :, random_height: random_height + img_rows, random_width: random_width + img_cols])

    if not unlabeled:
        return X_batch, y_batch
    return X_batch

class Batch_generator:

    def __init__(self, path, img_rows, img_cols, num_channels, cropping, horizontal_flip=False, vertical_flip=False, swap_axis=False, mask=-1, unlabeled=False):
        self.path = path
        self.f =  h5py.File(self.path, 'r')
        self.X_train = self.f['train']

        if not unlabeled:
            if mask == -1:  
                self.y_train = np.array(self.f['train_mask'])

            else:
                self.y_train = np.array(self.f['train_mask'])[:, mask]
                self.y_train = np.expand_dims(self.y_train, 1)

            print(self.y_train.shape)

        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.swap_axis = swap_axis

        self.num_channels = num_channels
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.cropping = cropping
        self.unlabeled = unlabeled

        # if not unlabeled:
        if mask == -1:
            # set this value as variable later
            self.num_mask_channels = 11
        else:
            self.num_mask_channels = 1

    def __call__(self, batch_size):

        if self.unlabeled:
            X_batch = form_batch(self.X_train, None, batch_size, self.num_channels, self.img_rows, self.img_cols, self.num_mask_channels, True)
        else:
            X_batch, y_batch = form_batch(self.X_train, self.y_train, batch_size, self.num_channels, self.img_rows, self.img_cols, self.num_mask_channels)

        for i in range(X_batch.shape[0]):
            xb = X_batch[i]
            if not self.unlabeled: 
                yb = y_batch[i]
   
            if self.horizontal_flip:
                if np.random.random() < 0.5:
                    xb = flip_axis(xb, 1)
                    if not self.unlabeled: 
                        yb = flip_axis(yb, 1)

            if self.vertical_flip:
                if np.random.random() < 0.5:
                    xb = flip_axis(xb, 2)
                    if not self.unlabeled: 
                        yb = flip_axis(yb, 2)

            if self.swap_axis:
                if np.random.random() < 0.5:
                    xb = xb.swapaxes(1, 2)
                    if not self.unlabeled: 
                        yb = yb.swapaxes(1, 2)

            X_batch[i] = xb
            if not self.unlabeled: 
                y_batch[i] = yb

        X_batch = np.transpose(X_batch, (0,2,3,1))
        if not self.unlabeled: 
            y_batch = np.transpose(y_batch, (0,2,3,1))
        
        # crop the labels if necessary
        if self.unlabeled: 
            return X_batch
        return X_batch, y_batch[:,self.cropping:self.img_rows-self.cropping,self.cropping:self.img_cols-self.cropping,:]

    def close(self):
        self.f.close()
