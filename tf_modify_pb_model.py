import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from tensorflow.python.platform import gfile
from sklearn.metrics import accuracy_score
from utils.mnist_reader import load_mnist

LESS_DATA = True
BATCH_SIZE = 100
EPOCHS = 2
LR = 1
PB_FILE = "output_models/tf_frozen_fashion.pb"


def main():
    x_train, y_train = load_mnist('data/fashion', kind='train')
    x_test, y_test = load_mnist('data/fashion', kind='t10k')
    x_train, y_train = shuffle(x_train, y_train)
    x_test, y_test = shuffle(x_test, y_test)

    if LESS_DATA:
        x_train, y_train = x_train[0:10000, ], y_train[0:10000, ]
        x_test, y_test = x_test[0:1000, ], y_test[0:1000, ]

    num_classes = len(set(y_train))

    n = x_train.shape[0]
    n_batches = n // BATCH_SIZE

    graph = tf.Graph()
    with graph.as_default():
        graph_def = tf.GraphDef()
        with gfile.FastGFile(PB_FILE, 'rb') as f:
            graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name="pb")
    # tf.summary.FileWriter("tensor_board/", graph=graph)

    graph_nodes = [n.name for n in graph.as_graph_def().node]

    print(graph_nodes)
    input_tensor = graph.get_tensor_by_name('pb/input:0')
    output_tensor = graph.get_tensor_by_name('pb/output_dense/BiasAdd:0')

    with tf.Session(graph=graph) as sess:
        output_result = sess.run(
            output_tensor, feed_dict={input_tensor: x_test})
        output_logit = np.argmax(output_result, axis=1)

        print("pb file result: {}".format(
            accuracy_score(y_test, output_logit)))

    # 任務：抽換dense layer 從128 個node 換成64 個node

    target_tensor = graph.get_tensor_by_name('pb/my_layer/dense/bias:0')

    with tf.Session(graph=graph) as sess:
        sess.run(tf.assign(target_tensor, np.zeros(128)))
    #     for i in range(EPOCHS):
    #         for j in range(n_batches):
    #             x_batch = x_train[j * BATCH_SIZE:(j * BATCH_SIZE + BATCH_SIZE)]
    #             y_batch = y_train[j * BATCH_SIZE:(j * BATCH_SIZE + BATCH_SIZE)]
    #
    #             sess.run(train_op, feed_dict={input_img: x_batch, labels: np.expand_dims(y_batch, axis=1)})
    #             test_cost, test_acc = sess.run([cost, acc_op],
    #                                            feed_dict={input_img: x_batch,
    #                                                       labels: np.expand_dims(y_batch, axis=1)})
    #
    #             print("Cost at epoch / iteration {}-{}: {:.3f}, acc: {:.3f}".format(i, j, test_cost, test_acc))




if __name__ == '__main__':
    main()
