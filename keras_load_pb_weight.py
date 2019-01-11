import keras
import numpy as np
import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D, Input, Flatten, Dense
from keras.models import Model
from sklearn.utils import shuffle
from tensorflow.python.framework import tensor_util
from tensorflow.python.platform import gfile

from utils import utils
from utils.mnist_reader import load_mnist

LESS_DATA = True
BATCH_SIZE = 100
EPOCHS = 2
LR = 1


def init_filter(shape):
    w = np.random.randn(*shape) * np.sqrt(2.0 / np.prod(shape[:-1]))
    return w.astype(np.float32)


def get_tensor_by_name(nodes, name):
    for node in nodes:
        if node.name == name:
            return node.attr['value'].tensor


def set_layer_weights(graph_nodes, keras_model, node_name, layer_name):
    weights = [tensor_util.MakeNdarray(get_tensor_by_name(graph_nodes, '{}/kernel'.format(node_name))),
               tensor_util.MakeNdarray(get_tensor_by_name(graph_nodes, '{}/bias'.format(node_name)))]
    layer = keras_model.get_layer(layer_name)
    layer.set_weights(weights)


def main():
    x_train, y_train = load_mnist('data/fashion', kind='train')
    x_test, y_test = load_mnist('data/fashion', kind='t10k')
    x_train, y_train = shuffle(x_train, y_train)
    x_test, y_test = shuffle(x_test, y_test)

    if LESS_DATA:
        x_train, y_train = x_train[0:10000, ], y_train[0:10000, ]
        x_test, y_test = x_test[0:10000, ], y_test[0:10000, ]

    num_classes = len(set(y_train))
    n = x_train.shape[0]
    n_batches = n // BATCH_SIZE

    # build tensorflow model

    input_img = tf.placeholder(tf.float32, shape=(None, 28, 28, 1), name='input')

    with tf.variable_scope('my_layer'):
        x = tf.layers.conv2d(input_img, filters=32, kernel_size=[3, 3], padding='same', activation=tf.nn.relu,
                             name='conv_1')
        x = tf.layers.conv2d(x, filters=64, kernel_size=[3, 3], padding='same', activation=tf.nn.relu, name='conv_2')
        x = tf.layers.max_pooling2d(x, pool_size=[2, 2], strides=2, padding='same', name='pool')
        x = tf.layers.flatten(x)
        x = tf.layers.dense(x, units=128, activation=tf.nn.relu, name='dense')
    output = tf.layers.dense(x, units=num_classes, name='output_dense')

    labels = tf.placeholder(tf.int32, shape=(None, 1), name='labels')  # 用tensorflow
    one_hot_labels = tf.one_hot(labels, num_classes)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=one_hot_labels, logits=output))
    train_op = tf.train.AdadeltaOptimizer(LR).minimize(cost)

    acc, acc_op = tf.metrics.accuracy(labels=labels,
                                      predictions=tf.argmax(output, 1))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        print([n.name for n in tf.get_default_graph().as_graph_def().node])
        # tf.summary.FileWriter("tensor_board/", graph=tf.get_default_graph())

        for i in range(EPOCHS):
            for j in range(n_batches):
                x_batch = x_train[j * BATCH_SIZE:(j * BATCH_SIZE + BATCH_SIZE)]
                y_batch = y_train[j * BATCH_SIZE:(j * BATCH_SIZE + BATCH_SIZE)]

                sess.run(train_op, feed_dict={input_img: x_batch, labels: np.expand_dims(y_batch, axis=1)})
                test_cost, test_acc = sess.run([cost, acc_op],
                                               feed_dict={input_img: x_batch,
                                                          labels: np.expand_dims(y_batch, axis=1)})

                print("Cost at epoch / iteration {}-{}: {:.3f}, acc: {:.3f}".format(i, j, test_cost, test_acc))

        test_cost, test_acc = sess.run([cost, acc],
                                       feed_dict={input_img: x_test,
                                                  labels: np.expand_dims(y_test, axis=1)})
        print("train complete! test cost: {:.3f}, acc: {:.3f}".format(test_cost, test_acc))

        # save model to pb file
        # tf.train.write_graph(sess.graph, "output_models/", "tf_fashion.pb", as_text=False)
        # tf.train.write_graph(sess.graph, "output_models/", "tf_fashion.pbtxt", as_text=True)

        # frozen graph
        frozen_graph = utils.freeze_session(sess)
        # tf.train.write_graph(frozen_graph, "output_models/", "tf_frozen_fashion.pb", as_text=False)
        # tf.train.write_graph(frozen_graph, "output_models/", "tf_frozen_fashion.pbtxt", as_text=True)

    # build keras model

    input_img = Input(shape=(28, 28, 1))

    x = Conv2D(32, (3, 3), padding='same', activation='relu', name='k_conv_1')(input_img)
    x = Conv2D(64, (3, 3), padding='same', activation='relu', name='k_conv_2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='k_pool')(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu', name='k_dense')(x)
    output = Dense(num_classes, name='k_output_dense')(x)

    keras_model = Model(inputs=input_img, outputs=output)
    keras_model.compile(loss='categorical_crossentropy',
                        optimizer=keras.optimizers.Adadelta(),
                        metrics=['accuracy'])

    one_hot_y_test = keras.utils.to_categorical(y_test, num_classes)
    score = keras_model.evaluate(x_test, one_hot_y_test, verbose=0)
    print('# Before set weights:')
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    with gfile.FastGFile("output_models/tf_frozen_fashion.pb", 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        graph_nodes = [n for n in graph_def.node]

    set_layer_weights(graph_nodes, keras_model, 'my_layer/conv_1', 'k_conv_1')
    set_layer_weights(graph_nodes, keras_model, 'my_layer/conv_2', 'k_conv_2')
    set_layer_weights(graph_nodes, keras_model, 'my_layer/dense', 'k_dense')
    set_layer_weights(graph_nodes, keras_model, 'output_dense', 'k_output_dense')

    score = keras_model.evaluate(x_test, one_hot_y_test, verbose=0)
    print('# After set weights:')
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


if __name__ == '__main__':
    main()
