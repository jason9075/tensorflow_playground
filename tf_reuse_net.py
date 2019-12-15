import cv2
import numpy  as np

import tensorflow as tf

INPUT_SHAPE = 100


def build_net(input_node, reuse=False):
    with tf.variable_scope('net', reuse=reuse):
        net = tf.layers.conv2d(input_node, 3, (5, 5))
        net = tf.layers.flatten(net)
        net = tf.layers.dense(net, 1)

    return net


def main():
    input_train = tf.placeholder(
        name='input_train',
        shape=[None, INPUT_SHAPE, INPUT_SHAPE, 3],
        dtype=tf.float32)

    input_valid = tf.placeholder(
        name='input_valid',
        shape=[None, INPUT_SHAPE, INPUT_SHAPE, 3],
        dtype=tf.float32)

    with tf.name_scope('train'):
        output_train = build_net(input_train)

    with tf.name_scope('val'):
        output_valid = build_net(input_valid, reuse=True)

    total_loss = output_train
    opt = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    grads = opt.compute_gradients(total_loss)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = opt.apply_gradients(grads)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        img = cv2.imread('data/kangaroo/k1.jpg')
        img = cv2.resize(img, (INPUT_SHAPE, INPUT_SHAPE))
        result = sess.run(output_train, feed_dict={input_train: np.expand_dims(img, axis=0)})
        print(result)
        result = sess.run(output_valid, feed_dict={input_valid: np.expand_dims(img, axis=0)})
        print(result)

        sess.run(train_op, feed_dict={input_train: np.expand_dims(img, axis=0)})

        result = sess.run(output_train, feed_dict={input_train: np.expand_dims(img, axis=0)})
        print(result)
        result = sess.run(output_valid, feed_dict={input_valid: np.expand_dims(img, axis=0)})
        print(result)

        tf.summary.FileWriter('tensor_board/', sess.graph)


if __name__ == '__main__':
    main()
