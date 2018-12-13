from tensorlayer.layers import *
import tensorflow as tf


def vgg16_model(x, is_train=True):
    n = 2
    act = tf.nn.relu
    n_filter = {'conv_11': 64, 'conv_12': 64, 'conv_21': 128, 'conv_22': 128, 'conv_31': 256, 'conv_32': 256,
                'conv_33': 256, 'conv_41': 512, 'conv_42': 512, 'conv_43': 512,
                'conv_51': 512, 'conv_52': 512, 'conv_53': 512}

    net_in = InputLayer(x, name='input')
    # conv1
    net = Conv2d(net_in, n_filter['conv_11'], filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME',
                 name='conv_11')
    net = BatchNormLayer(net, epsilon=1e-3, act=act, is_train=is_train, name='bn_11')
    net = Conv2d(net, n_filter=n_filter['conv_12'], filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME',
                 name='conv_12')
    net = BatchNormLayer(net, epsilon=1e-3, act=act, is_train=is_train, name='bn_12')
    net = MaxPool2d(net, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool1')

    # conv2
    net = Conv2d(net, n_filter=n_filter['conv_21'], filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME',
                 name='conv_21')
    net = BatchNormLayer(net, epsilon=1e-3, act=act, is_train=is_train, name='bn_21')
    net = Conv2d(net, n_filter=n_filter['conv_22'], filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME',
                 name='conv_22')
    net = BatchNormLayer(net, epsilon=1e-3, act=act, is_train=is_train, name='bn_22')
    net = MaxPool2d(net, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool2')

    # conv3
    net = Conv2d(net, n_filter=n_filter['conv_31'], filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME',
                 name='conv_31')
    net = BatchNormLayer(net, epsilon=1e-3, act=act, is_train=is_train, name='bn_31')
    net = Conv2d(net, n_filter=n_filter['conv_32'], filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME',
                 name='conv_32')
    net = BatchNormLayer(net, epsilon=1e-3, act=act, is_train=is_train, name='bn_32')
    net = Conv2d(net, n_filter=n_filter['conv_33'], filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME',
                 name='conv_33')
    net = BatchNormLayer(net, epsilon=1e-3, act=act, is_train=is_train, name='bn_33')
    net = MaxPool2d(net, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool3')

    # conv4
    net = Conv2d(net, n_filter=n_filter['conv_41'], filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME',
                 name='conv_41')
    net = BatchNormLayer(net, epsilon=1e-3, act=act, is_train=is_train, name='bn_41')
    net = Conv2d(net, n_filter=n_filter['conv_42'], filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME',
                 name='conv_42')
    net = BatchNormLayer(net, epsilon=1e-3, act=act, is_train=is_train, name='bn_42')
    net = Conv2d(net, n_filter=n_filter['conv_43'], filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME',
                 name='conv_43')
    net = BatchNormLayer(net, epsilon=1e-3, act=act, is_train=is_train, name='bn_43')
    net = MaxPool2d(net, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool4')

    # conv5
    net = Conv2d(net, n_filter=n_filter['conv_51'], filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME',
                 name='conv_51')
    net = BatchNormLayer(net, epsilon=1e-3, act=act, is_train=is_train, name='bn_51')
    net = Conv2d(net, n_filter=n_filter['conv_52'], filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME',
                 name='conv_52')
    net = BatchNormLayer(net, epsilon=1e-3, act=act, is_train=is_train, name='bn_52')
    net = Conv2d(net, n_filter=n_filter['conv_53'], filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME',
                 name='conv_53')
    net = BatchNormLayer(net, epsilon=1e-3, act=act, is_train=is_train, name='bn_53')

    # net = FlattenLayer(net, name='flatten')
    # net = DenseLayer(net, n_units=4096, act=tf.nn.relu, name='fc1_relu')
    # net = DenseLayer(net, n_units=4096, act=tf.nn.relu, name='fc2_relu')
    # net = DenseLayer(net, n_units=n, act=None, name='fc3_relu')

    net = GlobalMeanPool2d(net)
    net = DenseLayer(net, n_units=n, act=None, name='fc3_identity')
    return net.outputs, net


def small_model(x, is_train=True):
    act = tf.nn.relu
    n = 2

    network = InputLayer(x, name='input')
    network = Conv2dLayer(network, act=act, shape=(5, 5, 3, 32), strides=(1, 1, 1, 1), padding='SAME', name='conv_1')
    network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool_1')
    network = Conv2dLayer(network, act=act, shape=(5, 5, 32, 64), strides=(1, 1, 1, 1), padding='SAME', name='conv_2')
    network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool_2')
    network = Conv2dLayer(network, act=act, shape=(5, 5, 64, 128), strides=(1, 1, 1, 1), padding='SAME', name='conv_3')
    network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool_3')
    network = Conv2dLayer(network, act=act, shape=(5, 5, 128, 256), strides=(1, 1, 1, 1), padding='SAME', name='conv_4')
    network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool_4')

    network = FlattenLayer(network, name='flatten_layer')
    # TODO:maybe something wrong with dropout's fix
    network = DropoutLayer(network, keep=0.5, is_fix=is_train, name='drop_1')
    network = DenseLayer(network, n_units=256, act=tf.nn.relu, name='dense_1')
    network = DropoutLayer(network, keep=0.5, is_fix=is_train, name='drop_2')
    network = DenseLayer(network, n_units=n, act=tf.identity, name='dense_2')
    y = network.outputs
    return y, network


if __name__ == '__main__':
    input_pb = tf.placeholder(tf.float32, [None, 224, 224, 3])
    label_pb = tf.placeholder(tf.int32, [None])
    logist, net = vgg16_model(input_pb)
    exit()
