import tensorflow as tf
import numpy as np

from data import read_data, data_generator
from model import model, model2

# a = np.load('./matrix/fc2/0.npy')
b = np.load('./matrix/fc2.npy')
c = np.load('./matrix/final.npy')
temp = np.expand_dims(c[:, 0], axis=1)
mat = np.concatenate((temp, b), axis=1)
mat = mat.T

x_train, y_train, x_valid, y_valid, x_test, y_test = read_data('D:/DeepLearning/data/LongWoodCutPickJpg/')

input_pb = tf.placeholder(tf.float32, [None, 224, 224, 3])
logist, net = model(input_pb)

saver = tf.train.Saver()
with tf.Session() as sess:
    if tf.train.get_checkpoint_state('./ckpt/'):  # 确认是否存在
        saver.restore(sess, './ckpt/' + "ep050-step3000-loss0.000")
        print("load ok!")
    else:
        print("ckpt文件不存在")
    data_yield = data_generator(x_train, y_train)
    i = 0
    fc2_temp = None
    final_temp = None
    for img, lable in data_yield:
        b = net.all_layers
        # c = net.all_params
        # fc2_w = sess.run(net.all_params[-2])
        fc2_out, final_out = sess.run([net.all_layers[-2], net.all_layers[-1]], feed_dict={input_pb: img})
        np.save('./matrix/fc2/' + str(i), fc2_out)
        np.save('./matrix/final/' + str(i), final_out)
        if i == 0:
            fc2_temp = fc2_out
            final_temp = final_out
        else:
            fc2_temp = np.concatenate((fc2_temp, fc2_out), axis=0)
            final_temp = np.concatenate((final_temp, final_out), axis=0)
        i += 1
    np.save('./matrix/fc2.npy', fc2_temp)
    np.save('./matrix/final.npy', final_temp)
    exit()
