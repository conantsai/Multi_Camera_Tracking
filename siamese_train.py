from __future__ import generators, division, absolute_import, with_statement, print_function, unicode_literals

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage import io
from PIL import Image
from tqdm import tqdm

from siamese_dataset import Dataset
from siamese_model import *

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('batch_size', 64, 'Batch size.')
flags.DEFINE_integer('train_iter', 1000, 'Total training iter')
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
    left_datapath = "iLIDS-VID_train\i-LIDS-VID\sequences4\cam1"
    right_datapath = "iLIDS-VID_train\i-LIDS-VID\sequences4\cam2"
    left_images, left_labels = get_data(left_datapath)
    right_images, right_labels = get_data(right_datapath)

    model = siamese

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
            
			
        saver.save(sess, "model/model.ckpt")

## siamese2
# if __name__ == "__main__":
#     tf.reset_default_graph()
	
#     #setup dataset
#     left_datapath = "iLIDS-VID_train\i-LIDS-VID\sequences\cam1"
#     right_datapath = "iLIDS-VID_train\i-LIDS-VID\sequences\cam2"
#     left_images, left_labels = get_data(left_datapath)
#     right_images, right_labels = get_data(right_datapath)

#     model = siamese2

#     with tf.variable_scope('input_x1') as scope:
#         x_input_1 = tf.placeholder(tf.float32, shape=[None, data_hight, data_width, data_channel])
#     with tf.variable_scope('input_x2') as scope:
#         x_input_2 = tf.placeholder(tf.float32, shape=[None, data_hight, data_width, data_channel])
#     with tf.variable_scope('y') as scope:
#         y = tf.placeholder(tf.float32, shape=[None, 1])

#     with tf.name_scope('keep_prob') as scope:
#         keep_prob = tf.placeholder(tf.float32)

#     with tf.variable_scope('siamese') as scope:
#         out1 = model(x_input_1)
#         scope.reuse_variables()
#         out2 = model(x_input_2)
#     with tf.variable_scope('metrics') as scope:
#         loss = contrastive_loss(out1, out2, y, margin=0.5)
#         optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)

#     loss_summary = tf.summary.scalar('loss', loss)
#     merged_summary = tf.summary.merge_all()

#     dataset_group = Dataset(left_images, left_labels, right_images, right_labels)

# 	# Start Training
#     saver = tf.train.Saver()
#     with tf.Session() as sess:
#         sess.run(tf.global_variables_initializer())
#         sess.run(tf.local_variables_initializer())
#         writer = tf.summary.FileWriter('train.log', sess.graph)

# 		#train iter
#         for i in range(FLAGS.train_iter):
#             # batch_left, batch_right, label1, label2 = next_batch(FLAGS.batch_size, left_image, right_image, left_label, right_label)
#             # similarity_label = np.array(label1==label2, dtype=np.int)
#             # print(similarity_label)
#             left_image_group, left_label_group, right_image_group, right_label_group = dataset_group.next_batch(5)
#             dataset = Dataset(np.concatenate(left_image_group, axis=0), np.concatenate(left_label_group, axis=0), 
#                                 np.concatenate(right_image_group, axis=0), np.concatenate(right_label_group, axis=0))
#             left_image, left_label, right_image, right_label = dataset.next_batch(FLAGS.batch_size)

#             similarity_label = list()
#             for j in range(len(left_label)):
#                 if left_label[j] == right_label[j]:
#                     similarity_label.append([1])
#                 else:
#                     similarity_label.append([0])
#             similarity_label = np.array(similarity_label, dtype=int)

#             _, train_loss, summ = sess.run([optimizer, loss, merged_summary], feed_dict={x_input_1:left_image, x_input_2:right_image, y:similarity_label})
			
#             writer.add_summary(summ, i)
#             print("\r#%d - Loss"%i, train_loss)
			
#             # if (i + 1) % FLAGS.step == 0:
# 			# 	#generate test
# 			# 	# TODO: create a test file and run per batch
#             #     feat = sess.run(left_output, feed_dict={left:batch_left})
				
#             #     labels = batch_similarity
#             #     # plot result
#             #     f = plt.figure(figsize=(16,9))
#             #     f.set_tight_layout(True)
#             #     print(feat[labels==0, 0])
#             #     for j in range(10):
#             #         plt.plot(feat[labels==j, 0].flatten(), feat[labels==j, 1].flatten(), '.', c=colors[j], alpha=0.8)
#             #     plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
#             #     # plt.savefig('img/%d.jpg' % (i + 1))
#             #     plt.show()
#         saver.save(sess, "model/model.ckpt")





