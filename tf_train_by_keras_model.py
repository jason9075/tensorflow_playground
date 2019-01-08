# dataset: https://github.com/zalandoresearch/fashion-mnist
# https://blog.keras.io/keras-as-a-simplified-interface-to-tensorflow-tutorial.html

import keras
import tensorflow as tf
from keras import backend as K
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from sklearn.utils import shuffle

from utils.mnist_reader import load_mnist

LESS_DATA = True
BATCH_SIZE = 100
EPOCHS = 2
LR = 1


def main():
    sess = tf.Session()
    K.set_session(sess)

    x_train, y_train = load_mnist('data/fashion', kind='train')
    x_test, y_test = load_mnist('data/fashion', kind='t10k')
    x_train, y_train = shuffle(x_train, y_train)
    x_test, y_test = shuffle(x_test, y_test)

    if LESS_DATA:
        x_train, y_train = x_train[0:10000, ], y_train[0:10000, ]
        x_test, y_test = x_train[0:1000, ], y_train[0:1000, ]

    num_classes = len(set(y_train))

    # One-hot encoding
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    input_img = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))  # 用tensorflow

    x = Conv2D(32, kernel_size=(3, 3), activation='relu')(input_img)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(num_classes)(x)  # 不要activation 後面softmax_cross_entropy_with_logits_v2 會做

    n = x_train.shape[0]
    n_batches = n // BATCH_SIZE

    labels = tf.placeholder(tf.float32, shape=(None, num_classes))  # 用tensorflow
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=output))
    train_op = tf.train.AdadeltaOptimizer(LR).minimize(cost)

    acc, acc_op = tf.metrics.accuracy(labels=tf.argmax(labels, 1),
                                      predictions=tf.argmax(output, 1))

    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    for i in range(EPOCHS):
        for j in range(n_batches):
            x_batch = x_train[j * BATCH_SIZE:(j * BATCH_SIZE + BATCH_SIZE)]
            y_batch = y_train[j * BATCH_SIZE:(j * BATCH_SIZE + BATCH_SIZE)]

            sess.run(train_op, feed_dict={input_img: x_batch,
                                          labels: y_batch,
                                          K.learning_phase(): 1})
            test_cost, test_acc = sess.run([cost, acc_op],
                                           feed_dict={input_img: x_batch,
                                                      labels: y_batch})
            print("Cost at epoch / iteration {}-{}: {:.3f}, acc: {:.3f}".format(i, j, test_cost, test_acc))

    test_cost, test_acc = sess.run([cost, acc],
                                   feed_dict={input_img: x_test,
                                              labels: y_test,
                                              K.learning_phase(): 0})
    print("test cost: {:.3f}, acc: {:.3f}".format(test_cost, test_acc))


if __name__ == '__main__':
    main()
