from __future__ import generators, division, absolute_import, with_statement, print_function, unicode_literals

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.contrib.slim as slim

flags = tf.app.flags
FLAGS = flags.FLAGS

def siamese(input, is_training, reuse):
    if is_training:
        keep_prob = 0.5
    else:
        keep_prob = 1.0
    with tf.name_scope("model"):
        with tf.variable_scope("conv1") as scope:
            net = tf.contrib.layers.conv2d(inputs=input, num_outputs=32, kernel_size=[7, 7], stride=[1, 1], activation_fn=tf.nn.relu, padding='SAME',
                                            weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(), scope=scope, reuse=reuse)
            net = tf.contrib.layers.max_pool2d(inputs=net, kernel_size=[2, 2], padding='SAME')
            # net = tf.contrib.layers.dropout(net, keep_prob=keep_prob)
        print(net)

        with tf.variable_scope("conv2") as scope:
            net = tf.contrib.layers.conv2d(inputs=net, num_outputs=64, kernel_size=[5, 5], stride=[1, 1], activation_fn=tf.nn.relu, padding='SAME',
                                            weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(), scope=scope, reuse=reuse)
            net = tf.contrib.layers.max_pool2d(inputs=net, kernel_size=[2, 2], padding='SAME')
            # net = tf.contrib.layers.dropout(inputs=net, keep_prob=keep_prob)
        print(net)

        with tf.variable_scope("conv3") as scope:
            net = tf.contrib.layers.conv2d(inputs=net, num_outputs=128, kernel_size=[3, 3], stride=[1, 1], activation_fn=tf.nn.relu, padding='SAME',
                                            weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(), scope=scope, reuse=reuse)
            net = tf.contrib.layers.max_pool2d(inputs=net, kernel_size=[2, 2], padding='SAME')
            # net = tf.contrib.layers.dropout(inputs=net, keep_prob=keep_prob)
        print(net)

        with tf.variable_scope("conv4") as scope:
            net = tf.contrib.layers.conv2d(inputs=net, num_outputs=256, kernel_size=[3, 3], stride=[2, 2], activation_fn=tf.nn.relu, padding='SAME',
                                            weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(), scope=scope, reuse=reuse)
            net = tf.contrib.layers.max_pool2d(inputs=net, kernel_size=[2, 2], padding='SAME')
            # net = tf.contrib.layers.dropout(inputs=net, keep_prob=keep_prob)
        print(net)

        with tf.variable_scope("dense1") as scope:
            net = tf.contrib.layers.fully_connected(inputs=net, num_outputs=4, activation_fn=tf.nn.relu, scope=scope, reuse=reuse)
            # net = tf.contrib.layers.dropout(inputs=net, keep_prob=keep_prob)
        print(net)

        dense_size = int(net.shape[1])*int(net.shape[2])*int(net.shape[3])
        with tf.variable_scope("flatten1") as scope:
            # net = tf.contrib.layers.flatten(net)
            net = tf.reshape(tensor=net, shape=[-1, dense_size])
        print(net)
        
        return net


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

