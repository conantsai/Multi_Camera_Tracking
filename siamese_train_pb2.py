from __future__ import generators, division, absolute_import, with_statement, print_function, unicode_literals

import numpy as np
import matplotlib.pyplot as plt
import os
from skimage import io
from PIL import Image
from tqdm import tqdm

import tensorflow as tf
from tensorflow.python.framework import graph_util
import tensorflow.contrib.slim as slim

from siamese_dataset import Dataset
from siamese_model2 import *

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('batch_size', 64, 'Batch size.')
flags.DEFINE_integer('train_iter', 100, 'Total training iter')
flags.DEFINE_integer('step', 50, 'Save after ... iteration')
flags.DEFINE_string('model', 'mnist', 'model to run')

data_hight = 128
data_width = 64
data_channel = 3

right_id = list()
left_id = list()

def get_data(datapath):
    print('Getting images & labels ... ')

    images, labels = list(), list()
    for root, dirs, files in os.walk(datapath):
        if len(files) == 0: continue
        root = root.replace("\\", "/")
        image, label = list(), list()
        for index, content in enumerate(files):
            try:
                image_value = io.imread(root + "/" + content)
                image.append(image_value / 255)
                label.append(content.split("person")[1].split("_")[0])
            except OSError as e:
                # print(root + "/" + content)
                continue
        images.append(np.array(image))
        labels.append(np.array(label, dtype=int))

    # plt.imshow(left_image[1520])
    # plt.show()
    # print(left_label[1520])

    return np.array(images), np.array(labels)

if __name__ == "__main__":
    tf.reset_default_graph()
	
    #setup dataset
    left_datapath = "iLIDS-VID_train/i-LIDS-VID/sequences4/cam1"
    right_datapath = "iLIDS-VID_train/i-LIDS-VID/sequences4/cam2"
    left_images, left_labels = get_data(left_datapath)
    right_images, right_labels = get_data(right_datapath)

    model = siamese2

    x_input_1 = tf.placeholder(tf.float32, shape=[None, data_hight, data_width, data_channel], name='left')
    x_input_2 = tf.placeholder(tf.float32, shape=[None, data_hight, data_width, data_channel], name='right')
    y = tf.placeholder(tf.int32, shape=[None, 1], name='label')
    y = tf.to_float(y)

    out1 = model(x_input_1, is_training=True, reuse=False)
    out2 = model(x_input_2, is_training=True, reuse=True)

    loss = contrastive_loss(out1, out2, y, margin=0.5)

    global_step = tf.Variable(0, trainable=False)
    train_step = tf.train.MomentumOptimizer(0.01, 0.99, use_nesterov=True).minimize(loss, global_step=global_step)


    dataset_group = Dataset(left_images, left_labels, right_images, right_labels)

	# Start Training
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        #setup tensorboard	
        tf.summary.scalar('step', global_step)
        tf.summary.scalar('loss', loss)
        # for var in tf.trainable_variables():
        #     tf.summary.histogram(var.op.name, var)
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter('train.log', sess.graph)

		#train iter
        for i in range(FLAGS.train_iter):
            left_image_group, left_label_group, right_image_group, right_label_group = dataset_group.next_batch(5)
            dataset = Dataset(np.concatenate(left_image_group, axis=0), np.concatenate(left_label_group, axis=0), 
                                np.concatenate(right_image_group, axis=0), np.concatenate(right_label_group, axis=0))
            left_image, left_label, right_image, right_label = dataset.next_batch(FLAGS.batch_size)

            similarity_label = list()
            for j in range(len(left_label)):
                if left_label[j] == right_label[j]:
                    similarity_label.append([1])
                else:
                    similarity_label.append([0])
            similarity_label = np.array(similarity_label, dtype=int)

            _, l, summary_str = sess.run([train_step, loss, merged], feed_dict={x_input_1:left_image, x_input_2:right_image, y:similarity_label})
			
            writer.add_summary(summary_str, i)
            print("\r#%d - Loss"%i, l)

        constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['model/fc5/relu'])

        with tf.gfile.GFile('model/model.pb', mode='wb') as f:
            f.write(constant_graph.SerializeToString())
			
        saver.save(sess, "model/model.ckpt")

        # for node in sess.graph_def.node:
        #     print(node)


       




