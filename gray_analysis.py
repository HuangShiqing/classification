import tensorflow as tf
import numpy as np
import copy

from data import read_data, data_generator
from model import model, model2
import matplotlib.pyplot as plt

# a = np.load('./matrix/fc2/0.npy')
b = np.load('./matrix/fc2.npy')
c = np.load('./matrix/final.npy')
mat_1 = np.expand_dims(c[:, 0], axis=1)
mat_1 = np.concatenate((mat_1, b), axis=1)
mat_1 = mat_1.T

mat_2 = np.expand_dims(c[:, 1], axis=1)
mat_2 = np.concatenate((mat_2, b), axis=1)
mat_2 = mat_2.T

for p in range(2):
    if p == 0:
        mat = mat_1
    else:
        mat = mat_2
    # 2 无量纲化
    i = 0
    for i in range(len(mat)):
        mat[i] = mat[i] / max(mat[i])
    # 4 计算|x0-xi|
    j = 0
    for j in range(1, len(mat)):
        mat[j] = abs(mat[j] - mat[0])
    # 5 求最值
    min_num = 0
    max_num = 0
    l = 0
    for l in range(1, len(mat)):
        min_temp = np.min(mat[l])
        max_temp = np.max(mat[l])
        if min_temp < min_num:
            min_num = min_temp
        if max_temp > max_num:
            max_num = max_temp
    # 6 计算关联系数
    k = 0
    for k in range(1, len(mat)):
        mat[k] = (min_num + 0.5 * max_num) / (mat[k] + max_num * 0.5)
    # 7 计算每个指标的关联度
    final = np.mean(mat, axis=1)[1:]
    sort = np.sort(final)
    arg_sort = np.argsort(final)
    new = np.zeros([4096])
    n = 0
    for n in range(4096):
        new[arg_sort[n]] += 0.5 * n

final_argsort = np.argsort(new)
x = range(4096)
y = final_argsort
plt.scatter(x, y)
plt.imshow()


# x = range(1, 4096)
# plt.scatter(x, final[1:])
# plt.imshow()

x = list()
y = list()
for m in range(1, 1000):
    x.append(arg_sort[m])
    y.append(final[arg_sort[m]])
# x = range(1, 4097)
# y = final[1:]

# plt.ylim(0.865, 0.873)
plt.scatter(x, y)
plt.show()

x_train, y_train, x_valid, y_valid, x_test, y_test = read_data('/home/hsq/DeepLearning/data/LongWoodCutPickJpg/')

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
