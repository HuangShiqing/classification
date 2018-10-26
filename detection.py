import tensorflow as tf
import numpy as np

from data import read_data, data_generator
from model import model, model2

x_train, y_train, x_valid, y_valid, x_test, y_test = read_data('D:/DeepLearning/data/LongWoodCutPickJpg/')

input_pb = tf.placeholder(tf.float32, [None, 224, 224, 3])
label_pb = tf.placeholder(tf.int32, [None])
logist = model2(input_pb)

saver = tf.train.Saver()
with tf.Session() as sess:
    if tf.train.get_checkpoint_state('./ckpt/'):  # 确认是否存在
        saver.restore(sess, './ckpt/' + "ep110-step6500-loss0.000")
        print("load ok!")
    else:
        print("ckpt文件不存在")

    data_yield = data_generator(x_valid, y_valid)
    error_num = 0
    i = 0
    for img, lable in data_yield:
        logist_out = sess.run(logist, feed_dict={input_pb: img})
        logist_out = np.argmax(logist_out, axis=-1)
        a = np.equal(logist_out, lable)
        a = list(a)
        error_num += a.count(False)
        i += 1
    exit()