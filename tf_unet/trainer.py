import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug
import tensorlayer as tl
import logging	
import os
import shutil
import datetime
import cv2
import queue
import sys
import threading

# TODO remove the need for importing extra functions here
from extra_functions import make_prediction, evaluate_accuracy
from tf_unet.Pi_model import ramp_up_function
class Trainer(object):
    """
    Trains a unet instance

    :param net: the unet instance to train
    :param batch_size: size of training batch
    :param verification_batch_size: size of verification batch
    :param norm_grads: (optional) true if normalized gradients should be added to the summaries
    :param optimizer: (optional) name of the optimizer to use (momentum or adam)
    :param opt_kwargs: (optional) kwargs passed to the learning rate (momentum opt) and to the optimizer

    """



    def __init__(self, net, batch_size=1, verification_batch_size = 4, norm_grads=False, optimizer="momentum", mask=-1, test_x=None, test_y=None, save_image_dims=None, opt_kwargs={}):
        self.net = net
        self.batch_size = batch_size
        self.verification_batch_size = verification_batch_size
        self.norm_grads = norm_grads
        self.optimizer = optimizer
        self.mask = mask
        self.test_x = test_x
        self.test_y = test_y
        self.save_image_dims = save_image_dims
        self.opt_kwargs = opt_kwargs

    def get_data(self, sess, data_provider, unlabeled_data_provider=None):

        while True:
            batch_x, batch_y = data_provider(self.batch_size)

            if unlabeled_data_provider != None:
                unlabel_batch_x = unlabeled_data_provider(self.batch_size)
                sess.run(self.net.equ_x, feed_dict={self.net.u_x_shape: unlabel_batch_x})

            sess.run(self.net.eqx, feed_dict={self.net.x_shape: batch_x})
            sess.run(self.net.eqy, feed_dict={self.net.y_shape: batch_y})
            

        return

    # this let us get more control over optimisation, get this into param list later
    def _get_optimizer(self, training_iters, global_step):

        if self.optimizer == "momentum":
            learning_rate = self.opt_kwargs.pop("learning_rate", 0.2)
            decay_rate = self.opt_kwargs.pop("decay_rate", 0.5)
            momentum = self.opt_kwargs.pop("momentum", 0.2)
            steps  = self.opt_kwargs.pop("steps", 25)

            #self.learning_rate_node = tf.Variable(learning_rate, name="learning_rate")
            # define exponential learning rate decay
            self.learning_rate_node = tf.train.exponential_decay(learning_rate=learning_rate,
                                                                 global_step=global_step,
                                                                 decay_steps=training_iters*steps,
                                                                 decay_rate=decay_rate,
                                                                 staircase=True)

            optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate_node, momentum=momentum,
                                                   **self.opt_kwargs).minimize(self.net.cost,
                                                                               global_step=global_step)
        elif self.optimizer == "adam":
            learning_rate = self.opt_kwargs.pop("learning_rate", 0.001)
            self.learning_rate_node = tf.Variable(learning_rate, name="learning_rate")

            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_node,
                                               **self.opt_kwargs).minimize(self.net.cost,
                                                                           global_step=global_step)

        return optimizer

    def _initialize(self, training_iters, output_path, restore, prediction_path):

        global_step = tf.Variable(0, name="global_step")

        # (currently not used, need to learn more about norm gradient)
        #self.norm_gradients_node = tf.Variable(tf.constant(0.0, shape=[len(self.net.gradients_node)]), name="norm_gradients")
        #if self.net.summaries and self.norm_grads:
        #    tf.summary.histogram('norm_grads', self.norm_gradients_node)

        tf.summary.scalar('loss', self.net.cost)
        tf.summary.scalar('cross_entropy', self.net.cross_entropy)
        tf.summary.scalar('jaccard', self.net.jaccard)
        tf.summary.scalar('overall_accuracy', self.net.accuracy)
        #tf.summary.scalar('Accuracy_by_Class', self.net.IOU_by_Class)

        # initialise the optimizer
        self.optimizer = self._get_optimizer(training_iters, global_step)
        tf.summary.scalar('learning_rate', self.learning_rate_node)

        self.summary_op = tf.summary.merge_all()

        self.predicted_image = tf.placeholder("uint8", shape=[self.net.n_class, self.save_image_dims[0], self.save_image_dims[1], 1], name="pred")
        self.summary_image = tf.summary.image('Prediction', self.predicted_image, max_outputs=11)

        self.prediction_path = prediction_path

        # managing output/prediction directoies
        if not restore:
            logging.info("Removing '{:}'".format(output_path))
            shutil.rmtree(output_path, ignore_errors=True)

        if prediction_path != None:
            if not os.path.exists(prediction_path):
                logging.info("Allocating '{:}'".format(prediction_path))
                os.makedirs(prediction_path)

        if not os.path.exists(output_path):
            logging.info("Allocating '{:}'".format(output_path))
            os.makedirs(output_path)

        return

    def train(self, data_provider, unlabeled_data_provider, output_path, training_iters=10, epochs=100, dropout=0.75, display_step=1,
              restore=False, write_graph=False, prediction_path='./prediction', gpu_id="0", extra_unlabeled_provider=False):
        """
        Lauches the training process

        :param data_provider: callable returning training and verification data
        :param output_path: path where to store checkpoints
        :param training_iters: number of training mini batch iteration
        :param epochs: number of epochs
        :param dropout: dropout probability
        :param display_step: number of steps till outputting stats
        :param restore: Flag if previous model should be restored
        :param write_graph: Flag if the computation graph should be written as protobuf file to the output path
        :param prediction_path: path where to save predictions on each epoch
        """
 
        # initialise functions for tensorboard and output paths
        self._initialize(training_iters, output_path, restore, prediction_path)

        # enqueue & dequeue
        # q1 = tf.FIFOQueue(capacity=epochs*training_iters, dtypes=tf.float32)
        # q2 = tf.FIFOQueue(capacity=epochs*training_iters, dtypes=tf.float32)
        # batch_x, batch_y = data_provider(self.batch_size)

        # if extra_unlabeled_provider:
        #     unlabel_batch_x = unlabeled_data_provider(self.batch_size)

        # enqueue_x = q1.enqueue(batch_x)
        # enqueue_y = q2.enqueue(batch_y)
        # data_x = q1.dequeue()
        # data_y = q2.dequeue()

        # GPU config: put this to the main param list later
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        if gpu_id != "all":
            config.gpu_options.visible_device_list = gpu_id

        # initialise session and global variables
        sess = tf.Session(config=config)
        #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        tl.layers.initialize_global_variables(sess)
        #self.net.save(sess, output_path)

        # check later if write_graph is necessary
        if write_graph:
            tf.train.write_graph(sess.graph_def, output_path, "graph.pb", False)

        # restore model for continue training
        if restore:
            self.net.restore(sess, output_path + 'highest_model.npz')

        summary_writer = tf.summary.FileWriter(output_path, graph=sess.graph)
        logging.info("Start optimization")

        # Queue for storing data
        self.data_Queue = queue.Queue(10)

        # Thread
        if extra_unlabeled_provider:
            t1 = threading.Thread(target=self.get_data, args=(sess, data_provider, unlabeled_data_provider))
        else:
            t1 = threading.Thread(target=self.get_data, args=(sess, data_provider))
        t1.start()

        avg_gradients = None
        highest_IOU = 0
        for epoch in range(epochs):
            total_loss = 0
            now = datetime.datetime.now()
            self.net.ramp = ramp_up_function(epoch, 80)
            for step in range((epoch * training_iters), ((epoch + 1) * training_iters)):

                # load data
                #batch_x, batch_y = data_provider(self.batch_size)

                # Run optimization op (backprop)
                summary_str, _, loss, lr, acc, jaccard, unsuper_loss = sess.run(
                        (self.summary_op, self.optimizer, self.net.cost, self.learning_rate_node, self.net.accuracy, self.net.jaccard, self.net.unsuper_loss))
                # Look into the code below (currently not used)
                #if self.net.summaries and self.norm_grads:
                #    avg_gradients = _update_avg_gradients(avg_gradients, gradients, step)
                #    norm_gradients = [np.linalg.norm(gradient) for gradient in avg_gradients]
                #    self.norm_gradients_node.assign(norm_gradients).eval()

                #later1 = datetime.datetime.now()
                #print(later1 - now1)
                total_loss += loss

                self.output_minibatch_stats(summary_writer, summary_str, step, loss, total_loss/(step%training_iters + 1), jaccard, acc, unsuper_loss)

            later = datetime.datetime.now()

            # print(later - now)
            # print((later - now)/training_iters)

            print(later - now)
            print((later - now)/training_iters)

            self.output_epoch_stats(epoch, total_loss, training_iters, lr)
            # self.net.save(sess, output_path)

            if (epoch + 1 ) % 100 == 0:
                self.net.save(sess, output_path + str(epoch + 1) + "_")

            # if ( epoch  ) % display_step == 0:
            avg_IOU = self.store_prediction(sess, "Epoch_%s" % epoch, prediction_path, summary_writer, step)

            if avg_IOU > highest_IOU:
                print("Saving Net ! ...")
                self.net.save(sess, output_path + 'highest_')
                highest_IOU = avg_IOU
            sys.stdout.flush()
        # release resources
        sess.close()
        logging.info("Optimization Finished!")

        return

    def store_prediction(self, sess, name, prediction_path, summary_writer, step):

        # Make Prediction
        print("Making predictions ...")
        predicted_masks = make_prediction(self.net, None, self.test_x, input_size=(self.net.img_rows, self.net.img_cols), crop=self.net.crop, num_channels=self.net.channels, num_masks=self.net.n_class, sess=sess)
        
        # Evaluate Accuracy
        print("Evaluate Accuracy ...")
        Prediction, avg_IOU = evaluate_accuracy(prediction_path, predicted_masks, None, self.test_y, mask=self.mask, epsilon=1e-12)

        # Save prediction
        print("Save prediction ...")
        for i in range(Prediction.shape[-1]):
            Prediction[0, :self.save_image_dims[0], :self.save_image_dims[1], i] = cv2.resize(Prediction[0,:,:,i], (self.save_image_dims[0], self.save_image_dims[1]), interpolation=cv2.INTER_CUBIC)

        summary_str = sess.run(self.summary_image, feed_dict={self.predicted_image: Prediction[:,:self.save_image_dims[0], :self.save_image_dims[1], :]})
        summary_writer.add_summary(summary_str, step)
        summary_writer.flush()
        return avg_IOU


    def output_epoch_stats(self, epoch, total_loss, training_iters, lr):
        logging.info("Epoch {:}, Average loss: {:.4f}, learning rate: {:.4f}".format(epoch, (total_loss / training_iters), lr))

    def output_minibatch_stats(self, summary_writer, summary_str, step, loss, avg_loss, jaccard, acc, unsuper_loss):

        summary_writer.add_summary(summary_str, step)
        summary_writer.flush()
        logging.info("Iter {:}, Minibatch Loss= {:.4f}, Average Loss= {:.4f}, Jaccard= {:.4f}, Accuracy= {:.4f}".format(step,loss,avg_loss,jaccard,acc, unsuper_loss))

