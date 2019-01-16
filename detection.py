import tensorflow as tf
import numpy as np

from data import read_data, data_generator
from model import *
from varible import *

x_train, y_train, x_valid, y_valid = read_data(Gb_data_dir)

input_pb = tf.placeholder(tf.float32, [None, 224, 224, 3])
label_pb = tf.placeholder(tf.int32, [None])
logist, net = vgg16_adjusted(input_pb, is_train=False)

saver = tf.train.Saver()
with tf.Session() as sess:
    try:
        saver.restore(sess, '/media/hsq/新加卷/ubuntu/ckpt/dogVScat/vgg16_adjusted/0/' + "ep450-step281500-loss0.001")
        print("load ok!")
    except:
        print("ckpt文件不存在")
        raise

    data_yield = data_generator(x_valid, y_valid, is_train=False)
    error_num = 0
    i = 0
    for img, lable in data_yield:
        logist_out = sess.run(logist, feed_dict={input_pb: img})
        logist_out = np.argmax(logist_out, axis=-1)
        a = np.equal(logist_out, list(map(int, lable)))
        a = list(a)
        error_num += a.count(False)
        i += 1
    print('error: ', str(error_num), ' in ', str(i * Gb_batch_size))
