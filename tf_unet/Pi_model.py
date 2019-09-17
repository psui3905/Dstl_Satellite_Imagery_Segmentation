import numpy as np
import logging
import os
import datetime

import tensorflow as tf
# import tensorlayer as tl
import math

from tf_unet import util
from tf_unet.loss import (cross_entropy,jaccard_coef_loss,jaccard_coef_int_avg,jaccard_coef_int,get_loss,new_jaccard)
from tf_unet.unet_model import unet


def temporal_ensembling_loss(X_train_labeled, y_train_labeled, X_train_unlabeled, model, unsupervised_weight, ensembling_targets):
    """ Gets the loss for the temporal ensembling model

    Arguments:
        X_train_labeled {tensor} -- labeled samples
        y_train_labeled {tensor} -- labeled train labels
        X_train_unlabeled {tensor} -- unlabeled samples 
        model {tf.keras.Model} -- temporal ensembling model
        unsupervised_weight {float} -- weight of the unsupervised loss
        ensembling_targets {np.array} --  ensembling targets

    Returns:
        {tensor} -- predictions for the ensembles
        {tensor} -- loss value
    """

    z_labeled = model.call(X_train_labeled)
    z_labeled_i = model.call(X_train_labeled)

    z_unlabeled = model.call(X_train_unlabeled)
    z_unlabeled_i = model.call(X_train_unlabeled)

    # current_predictions = tf.concat([z_labeled, z_unlabeled], 0)

    return tf.losses.softmax_cross_entropy(
        y_train_labeled, z_labeled) + unsupervised_weight * (
            tf.losses.mean_squared_error(ensembling_targets, current_predictions))


def temporal_ensembling_gradients(X_train_labeled, y_train_labeled, X_train_unlabeled, model, unsupervised_weight, ensembling_targets):
    """ Gets the gradients for the temporal ensembling model

    Arguments:
        X_train_labeled {tensor} -- labeled samples
        y_train_labeled {tensor} -- labeled train labels
        X_train_unlabeled {tensor} -- unlabeled samples 
        model {tf.keras.Model} -- temporal ensembling model
        unsupervised_weight {float} -- weight of the unsupervised loss
        ensembling_targets {np.array} --  ensembling targets

    Returns:
        {tensor} -- predictions for the ensembles
        {tensor} -- loss value
        {tensor} -- gradients for each model variables
    """

    with tf.GradientTape() as tape:
        ensemble_precitions, loss_value = temporal_ensembling_loss(X_train_labeled, y_train_labeled, X_train_unlabeled,
                                                                   model, unsupervised_weight, ensembling_targets)

    return ensemble_precitions, loss_value, tape.gradient(loss_value, model.variables)


# def pi_model_loss(X_train_labeled, y_train_labeled, pi_model_1, pi_model_2, unsupervised_weight):
#     """ Gets the Loss Value for SSL Pi Model

#     Arguments:
#         X_train_labeled {tensor} -- train images
#         y_train_labeled {tensor} -- train labels
#         X_train_unlabeled {tensor} -- unlabeled train images
#         pi_model {tf.keras.Model} -- model to be trained
#         unsupervised_weight {float} -- weight

#     Returns:
#         {tensor} -- loss value
#     """
#     z_labeled = pi_model_1()
#     z_labeled_i = pi_model_2()

#     # Loss = supervised loss + unsup loss of labeled sample + unsup loss unlabeled sample
#     return tf.losses.softmax_cross_entropy(y_train_labeled, z_labeled) + unsupervised_weight * (tf.losses.mean_squared_error(z_labeled, z_labeled_i))



def ramp_up_function(epoch, epoch_with_max_rampup=80):
    """ Ramps the value of the weight and learning rate according to the epoch
        according to the paper

    Arguments:
        {int} epoch
        {int} epoch where the rampup function gets its maximum value

    Returns:
        {float} -- rampup value
    """

    if epoch < epoch_with_max_rampup:
        p = max(0.0, float(epoch)) / float(epoch_with_max_rampup)
        p = 1.0 - p
        return math.exp(-p*p*5.0)
    else:
        return 1.0


def ramp_down_function(epoch, num_epochs):
    """ Ramps down the value of the learning rate and adam's beta
        in the last 50 epochs according to the paper

    Arguments:
        {int} current epoch
        {int} total epochs to train

    Returns:
        {float} -- rampup value
    """
    epoch_with_max_rampdown = 50

    if epoch >= (num_epochs - epoch_with_max_rampdown):
        ep = (epoch - (num_epochs - epoch_with_max_rampdown)) * 0.5
        return math.exp(-(ep * ep) / epoch_with_max_rampdown)
    else:
        return 1.0


class PiModel:
    """ Class for defining eager compatible tfrecords file

        I did not use tfe.Network since it will be depracated in the
        future by tensorflow.
    """

    def call(self, X, training=False, keep_prob=tf.placeholder(tf.float32, name="dropout_probability"), channels=3, batch_norm=False, n_class=1, img_rows=112, img_cols=112, is_train=True, cropping=0, cost_kwargs={"regularizer": None}, **kwargs):
        """ Function that allows running a tensor through the pi model

        Arguments:
            input {[tensor]} -- batch of images
            training {bool} -- if true applies augmentaton and additive noise

        Returns:
            [tensor] -- predictions
        """

        if training:
            h = self.__aditive_gaussian_noise(X, 0.15)
            # h = self.__apply_image_augmentation(h)
        else:
            h = X

        # create model 
        self.model = unet(X, keep_prob, channels, n_class, img_rows, img_cols, batch_norm, is_train=True, reuse=tf.AUTO_REUSE, **kwargs)

        # obtain logit from tensorlayer model
        # batch 16, 2 -- width, 3 -- height, channel -- 11
        self.logits = self.model.outputs[:,cropping:img_rows-cropping,cropping:img_cols-cropping,:]
        return [self.logits, self.model]

    def __aditive_gaussian_noise(self, input, std):
        """ Function to add additive zero mean noise as described in the paper

        Arguments:
            input {tensor} -- image
            std {int} -- std to use in the random_normal

        Returns:
            {tensor} -- image with added noise
        """

        noise = tf.random_normal(shape=tf.shape(input), mean=0.0, stddev=std, dtype=tf.float32)
        return input + noise

    def __apply_image_augmentation(self, image):
        """ Applies random transformation to the image

        Arguments:
            image {tensor} -- image

        Returns:
            {tensor} -- transformed image
        """

        random_shifts = np.random.randint(-2, 2, image.numpy().shape[0])
        random_transformations = tf.contrib.image.translations_to_projective_transforms(
            random_shifts)
        image = tf.contrib.image.transform(image, random_transformations, 'NEAREST',
                                           output_shape=tf.convert_to_tensor(image.numpy().shape[1:3], dtype=np.int32))
        return image

