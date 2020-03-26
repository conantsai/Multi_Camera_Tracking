import numpy as np
import matplotlib.pyplot as plt
import os
from skimage import io
from scipy.spatial.distance import cdist
from matplotlib import gridspec

from siamese_model import *
from siamese_dataset import *

import tensorflow as tf
from tensorflow.python.platform import gfile


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

    ## checkpoint
    img_placeholder = tf.placeholder(tf.float32, [None, 128, 64, 3], name='img')
    net = siamese(img_placeholder, is_training=False, reuse=False)

    idx = 0
    im = left_images[idx]
    im = np.expand_dims(im, axis = 0)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, "model/model.ckpt")
        left_feat = sess.run(net, feed_dict={img_placeholder:im}) 

    print(left_feat)

    ## pb 
    tf.reset_default_graph()
    sess = tf.Session()
    with tf.io.gfile.GFile('model/model.pb', 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')

    graph = tf.compat.v1.get_default_graph()
    predict_tensor = graph.get_tensor_by_name("model/flatten1/Reshape:0")
    input_tensor = graph.get_tensor_by_name("left:0")

    predict = sess.run(predict_tensor, feed_dict={input_tensor: im})

    print(predict)


if __name__ == "__main__":
    test()