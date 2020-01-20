from __future__ import generators, division, absolute_import, with_statement, print_function, unicode_literals

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.contrib.slim as slim

flags = tf.app.flags
FLAGS = flags.FLAGS

def mnist_model(input, reuse=False):
	with tf.name_scope("model"):
		with tf.variable_scope("conv1") as scope:
			net = tf.contrib.layers.conv2d(input, 32, [7, 7], activation_fn=tf.nn.relu, padding='SAME',
		        weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),scope=scope,reuse=reuse)
			net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')

		with tf.variable_scope("conv2") as scope:
			net = tf.contrib.layers.conv2d(net, 64, [5, 5], activation_fn=tf.nn.relu, padding='SAME',
		        weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),scope=scope,reuse=reuse)
			net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')

		with tf.variable_scope("conv3") as scope:
			net = tf.contrib.layers.conv2d(net, 128, [3, 3], activation_fn=tf.nn.relu, padding='SAME',
		        weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),scope=scope,reuse=reuse)
			net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')

		with tf.variable_scope("conv4") as scope:
			net = tf.contrib.layers.conv2d(net, 256, [1, 1], activation_fn=tf.nn.relu, padding='SAME',
		        weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),scope=scope,reuse=reuse)
			net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')

		with tf.variable_scope("conv5") as scope:
			net = tf.contrib.layers.conv2d(net, 2, [1, 1], activation_fn=None, padding='SAME',
		        weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),scope=scope,reuse=reuse)
			net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')

		net = tf.contrib.layers.flatten(net)
	
	return net

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

        net = tf.contrib.layers.flatten(net)
        print(net)
        
        return net

def siamese2(input):
    with tf.name_scope("model"):
        with tf.name_scope('conv1') as scope:
            w1 = tf.Variable(tf.truncated_normal(shape=[7,7,3,128], stddev=0.05), name='w1')
            b1 = tf.Variable(tf.zeros(128), name='b1')
            conv1 = tf.nn.conv2d(input, w1, strides=[1,1,1,1], padding='SAME', name='conv1')
        with tf.name_scope('relu1') as scope:
            relu1 = tf.nn.relu(tf.add(conv1, b1), name='relu1')
        with tf.name_scope('bn1') as scope:
            bn1 = tf.layers.batch_normalization(relu1, name='bn1')
        with tf.name_scope('pool1') as scope:
            pool1 = tf.nn.max_pool(bn1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='pool1')
        with tf.name_scope('drop1') as scope:
            drop1 = tf.nn.dropout(pool1, keep_prob=1.0, name='drop1')

        with tf.name_scope('conv2') as scope:
            w2 = tf.Variable(tf.truncated_normal(shape=[5,5,128,256], stddev=0.05), name='w2')
            b2 = tf.Variable(tf.zeros(256), name='b2')
            conv2 = tf.nn.conv2d(drop1, w2, strides=[1,1,1,1], padding='SAME', name='conv2')
        with tf.name_scope('relu2') as scope:
            relu2 = tf.nn.relu(tf.add(conv2, b2), name='relu2')
        with tf.name_scope('bn2') as scope:
            bn2 = tf.layers.batch_normalization(relu2, name='bn2')
        with tf.name_scope('pool2') as scope:
            pool2 = tf.nn.max_pool(bn2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='pool2')
        with tf.name_scope('drop2') as scope:
            drop2 = tf.nn.dropout(pool2, keep_prob=1.0, name='drop2')

        with tf.name_scope('conv3') as scope:
            w3 = tf.Variable(tf.truncated_normal(shape=[3,3,256,512], stddev=0.05), name='w3')
            b3 = tf.Variable(tf.zeros(512), name='b3')
            conv3 = tf.nn.conv2d(drop2, w3, strides=[1,1,1,1], padding='SAME', name='conv3')
        with tf.name_scope('relu3') as scope:
            relu3 = tf.nn.relu(tf.add(conv3, b3), name='relu3')
        with tf.name_scope('bn3') as scope:
            bn3 = tf.layers.batch_normalization(relu3, name='bn3')
        with tf.name_scope('pool3') as scope:
            pool3 = tf.nn.max_pool(bn3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='pool3')
        with tf.name_scope('drop3') as scope:
            drop3 = tf.nn.dropout(pool3, keep_prob=1.0, name='drop3')

        with tf.name_scope('conv4') as scope:
            w4 = tf.Variable(tf.truncated_normal(shape=[3,3,512,256], stddev=0.05), name='w4')
            b4 = tf.Variable(tf.zeros(256), name='b4')
            conv4 = tf.nn.conv2d(drop3, w4, strides=[1,2,2,1], padding='SAME', name='conv4')
        with tf.name_scope('relu4') as scope:
            relu4 = tf.nn.relu(tf.add(conv4, b4), name='relu4')
        with tf.name_scope('bn4') as scope:
            bn4 = tf.layers.batch_normalization(relu4, name='bn4')
        with tf.name_scope('pool4') as scope:
            pool4 = tf.nn.max_pool(bn4, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='pool4')
        with tf.name_scope('drop4') as scope:
            drop4 = tf.nn.dropout(pool4, keep_prob=1.0, name='drop4')
        
        with tf.name_scope('dense1') as scope:
            x_flat = tf.reshape(drop4, shape=[-1,int(drop4.shape[1]*drop4.shape[2]*drop4.shape[3])])
            w_dense1 = tf.Variable(tf.truncated_normal(shape=[int(drop4.shape[1]*drop4.shape[2]*drop4.shape[3]),16], stddev=0.05, mean=0), name='w_dense1')
            b_dense1 = tf.Variable(tf.zeros(16), name='b_dense1')
            dense1 = tf.add(tf.matmul(x_flat, w_dense1), b_dense1)
        with tf.name_scope('relu_dense1') as scope:
            relu_dense1 = tf.nn.relu(dense1, name='relu_fc2')
        with tf.name_scope('drop_dense1') as scope:
            drop_dense1 = tf.nn.dropout(relu_dense1, keep_prob=1.0, name='drop_dense1')

        with tf.name_scope('flatten1') as scope:
            net = tf.layers.flatten(drop_dense1)

        return net

def siamese3(input):
    with tf.name_scope("model"):
        with tf.name_scope('conv1') as scope:
            w1 = tf.Variable(tf.truncated_normal(shape=[7,7,3,128]), name='w1')
            b1 = tf.Variable(tf.zeros(128), name='b1')
            conv1 = tf.nn.conv2d(input, w1, strides=[1,1,1,1], padding='SAME', name='conv1')
        with tf.name_scope('relu1') as scope:
            relu1 = tf.nn.relu(tf.add(conv1, b1), name='relu1')
        with tf.name_scope('pool1') as scope:
            pool1 = tf.nn.max_pool(relu1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='pool1')
        
        with tf.name_scope('dense1') as scope:
            x_flat = tf.reshape(pool1, shape=[-1,int(pool1.shape[1]*pool1.shape[2]*pool1.shape[3])])
            w_dense1 = tf.Variable(tf.truncated_normal(shape=[int(pool1.shape[1]*pool1.shape[2]*pool1.shape[3]),16]), name='w_dense1')
            b_dense1 = tf.Variable(tf.zeros(16), name='b_dense1')
            dense1 = tf.add(tf.matmul(x_flat, w_dense1), b_dense1)
        with tf.name_scope('relu_dense1') as scope:
            relu_dense1 = tf.nn.relu(dense1, name='relu_fc2')

        with tf.name_scope('flatten1') as scope:
            net = tf.layers.flatten(relu_dense1)

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

