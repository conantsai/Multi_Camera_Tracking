import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage import io
from scipy.spatial.distance import cdist
from matplotlib import gridspec

from siamese_model import *
from siamese_dataset import *

#helper function to plot image
def show_image(idxs, data, label):
    if type(idxs) != np.ndarray:
        idxs = np.array([idxs])
    fig = plt.figure()
    gs = gridspec.GridSpec(1, len(idxs))
    for i in range(len(idxs)):
        ax = fig.add_subplot(gs[0, i])
        print(label[idxs[i]])
        ax.imshow(data[idxs[i], :, :, :])
        ax.set_title(label[idxs[i]])
        ax.axis('off')
    plt.show()

def test():
    left_datapath = "iLIDS-VID_test\i-LIDS-VID\images2\cam1"
    right_datapath = "iLIDS-VID_test\i-LIDS-VID\images2\cam2"
    left_images, left_labels = get_data_test(left_datapath, colorTrans=False)
    right_images, right_labels = get_data_test(right_datapath, colorTrans=False)

    left_images = left_images
    right_images = right_images

    len_left = len(left_images)
    len_right = len(right_images)

    img_placeholder = tf.placeholder(tf.float32, [None, 128, 64, 3], name='img')
    net = siamese(img_placeholder, is_training=True, reuse=False)
    # net = siamese2(img_placeholder)

    #generate new random test image
    # idx = np.random.randint(0, len_left)
    # print(idx)
    idx = 0
    im = left_images[idx]

    #show the test image
    # show_image(idx, left_images, left_labels)
    # print("This is image from id & label:", idx, left_labels[idx])

    #run the test image through the network to get the test features
    # tf.reset_default_graph()
    # saver = tf.train.import_meta_graph("model/model.ckpt.meta")
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, "model/model.ckpt")
        left_feat = sess.run(net, feed_dict={img_placeholder:[im]}) 

    print(left_feat)

    # # tf.reset_default_graph()
    # saver = tf.train.import_meta_graph("model/model.ckpt.meta")
    # # saver = tf.train.Saver()
    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #     ckpt = tf.train.get_checkpoint_state("model")
    #     saver.restore(sess, "model/model.ckpt")
    #     right_feat = sess.run(net, feed_dict={img_placeholder:right_images}) 

    # #calculate the cosine similarity and sort
    # dist = cdist(right_feat, left_feat, 'cosine')
    # print(dist)
    # rank = np.argsort(dist.ravel())

    #show the top n similar image from train data
    # n = 7
    # show_image(rank[:n], right_images, right_labels)
    # print("retrieved ids:", rank[:n], dist)



if __name__ == "__main__":
    test()