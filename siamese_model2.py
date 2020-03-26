from __future__ import generators, division, absolute_import, with_statement, print_function, unicode_literals

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.contrib.slim as slim
from neuralLayer import Layer

flags = tf.app.flags
FLAGS = flags.FLAGS

from tensorflow.python.eager import context as _context

def siamese2(input, is_training, reuse):
    siamese_layer = Layer()

    with tf.name_scope("model"):
        if reuse == False:
            conv1 = siamese_layer.Conv2D(input_op=input, output_op=32, kernel_height=7, kernel_width=7, stride_height=1, stride_width=1, norm_init_mean=0.0, 
                                                norm_init_stddev=0.05, const_init=0.0, padding="SAME", name="conv1")
            pool1 = siamese_layer.MaxPool2D(input_op=conv1, kernel_height=2, kernel_width=2, stride_height=2, stride_width=2, padding="SAME", name='maxpool1')
            conv2 = siamese_layer.Conv2D(input_op=pool1, output_op=64, kernel_height=5, kernel_width=5, stride_height=1, stride_width=1, norm_init_mean=0.0, 
                                                norm_init_stddev=0.05, const_init=0.0, padding="SAME", name="conv2")
            pool2 = siamese_layer.MaxPool2D(input_op=conv2, kernel_height=2, kernel_width=2, stride_height=2, stride_width=2, padding="SAME", name='maxpool2')
            conv3 = siamese_layer.Conv2D(input_op=pool2, output_op=128, kernel_height=3, kernel_width=3, stride_height=1, stride_width=1, norm_init_mean=0.0, 
                                                norm_init_stddev=0.05, const_init=0.0, padding="SAME", name="conv3")
            pool3 = siamese_layer.MaxPool2D(input_op=conv3, kernel_height=2, kernel_width=2, stride_height=2, stride_width=2, padding="SAME", name='maxpool3')
            conv4 = siamese_layer.Conv2D(input_op=pool3, output_op=256, kernel_height=3, kernel_width=3, stride_height=2, stride_width=2, norm_init_mean=0.0, 
                                                norm_init_stddev=0.05, const_init=0.0, padding="SAME", name="conv4")
            pool4 = siamese_layer.MaxPool2D(input_op=conv4, kernel_height=2, kernel_width=2, stride_height=2, stride_width=2, padding="SAME", name='maxpool4')

            fc5 = siamese_layer.FullyConnected(input_op=pool4, output_op=4, norm_init_mean=0.0, norm_init_stddev=0.01, const_init=0.0, activation_function="relu", 
                                                    name="fc5")
            
            dense_size = int(fc5.shape[1])*int(fc5.shape[2])*int(fc5.shape[3])
            flat6 = siamese_layer.Flatten(input_op=fc5, shape=[-1, dense_size], name="flat6")
        else:
            conv1 = siamese_layer.Conv2D_reuse(input_op=input, output_op=32, kernel_height=7, kernel_width=7, stride_height=1, stride_width=1, norm_init_mean=0.0, 
                                                norm_init_stddev=0.05, const_init=0.0, padding="SAME", name="conv1")
            pool1 = siamese_layer.MaxPool2D_reuse(input_op=conv1, kernel_height=2, kernel_width=2, stride_height=2, stride_width=2, padding="SAME", name='maxpool1')
            conv2 = siamese_layer.Conv2D_reuse(input_op=pool1, output_op=64, kernel_height=5, kernel_width=5, stride_height=1, stride_width=1, norm_init_mean=0.0, 
                                                norm_init_stddev=0.05, const_init=0.0, padding="SAME", name="conv2")
            pool2 = siamese_layer.MaxPool2D_reuse(input_op=conv2, kernel_height=2, kernel_width=2, stride_height=2, stride_width=2, padding="SAME", name='maxpool2')
            conv3 = siamese_layer.Conv2D_reuse(input_op=pool2, output_op=128, kernel_height=3, kernel_width=3, stride_height=1, stride_width=1, norm_init_mean=0.0, 
                                                norm_init_stddev=0.05, const_init=0.0, padding="SAME", name="conv3")
            pool3 = siamese_layer.MaxPool2D_reuse(input_op=conv3, kernel_height=2, kernel_width=2, stride_height=2, stride_width=2, padding="SAME", name='maxpool3')
            conv4 = siamese_layer.Conv2D_reuse(input_op=pool3, output_op=256, kernel_height=3, kernel_width=3, stride_height=2, stride_width=2, norm_init_mean=0.0, 
                                                norm_init_stddev=0.05, const_init=0.0, padding="SAME", name="conv4")
            pool4 = siamese_layer.MaxPool2D_reuse(input_op=conv4, kernel_height=2, kernel_width=2, stride_height=2, stride_width=2, padding="SAME", name='maxpool4')

            fc5 = siamese_layer.FullyConnected_reuse(input_op=pool4, output_op=4, norm_init_mean=0.0, norm_init_stddev=0.01, const_init=0.0, activation_function="relu", 
                                                    name="fc5")
            print(fc5)

            dense_size = int(fc5.shape[1])*int(fc5.shape[2])*int(fc5.shape[3])
            flat6 = siamese_layer.Flatten_reuse(input_op=fc5, shape=[-1, dense_size], name="flat6")

    return flat6


def contrastive_loss(model1, model2, y, margin):
    with tf.name_scope("contrastive-loss"):
        distance = tf.sqrt(tf.reduce_sum(tf.pow(model1 - model2, 2), 1, keepdims=True))
        similarity = y * tf.square(distance)                                           # keep the similar label (1) close to each other
        dissimilarity = (1 - y) * tf.square(tf.maximum((margin - distance), 0))        # give penalty to dissimilar label if the distance is bigger than margin
        return tf.reduce_mean(dissimilarity + similarity) / 2

def siamese_loss(out1, out2, y, Q=5):
    Q = tf.constant(Q, name="Q", dtype=tf.float32)
    E_w = tf.sqrt(tf.reduce_sum(tf.square(out1-out2), 1))  
    pos = tf.multiply(tf.multiply(y, 2/Q), tf.square(E_w))
    neg = tf.multiply(tf.multiply(1-y, 2*Q), tf.exp(-2.77/Q*E_w))                
    loss = pos + neg                 
    loss = tf.reduce_mean(loss)              
    return loss

