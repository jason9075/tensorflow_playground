import cv2
import numpy as np

import tensorflow as tf

SHAPE_W = 1000
SHAPE_H = 600
INPUT_IMAGE = 'data/kangaroo/k1.jpg'
CKPT_PATH = 'output_model/partition.ckpt'


def build_net(input_node):
    net = tf.layers.conv2d(input_node, 3, (5, 5))
    net = tf.layers.flatten(net)
    net = tf.layers.dense(net, 1)

    return net


def save():
    path_tensor = tf.placeholder(tf.string, shape=None, name="path")

    file_contents = tf.read_file(path_tensor)
    image = tf.image.decode_image(file_contents, channels=3)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    image.set_shape((SHAPE_W, SHAPE_H, 3))
    image = tf.expand_dims(image, axis=0)

    output = build_net(image)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        result = sess.run(output, feed_dict={path_tensor: INPUT_IMAGE})
        print(result)

        saver = tf.train.Saver()
        saver.save(sess, CKPT_PATH)


def load():
    input_layer = tf.placeholder(
        name='input_images',
        shape=[None, SHAPE_H, SHAPE_W, 3],
        dtype=tf.float32)

    output = build_net(input_layer)

    img = cv2.imread(INPUT_IMAGE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255

    with tf.Session() as sess:
        restore_saver = tf.train.Saver()
        restore_saver.restore(sess, CKPT_PATH)

        result = sess.run(output, feed_dict={input_layer: np.expand_dims(img, axis=0)})
        print(result)


if __name__ == '__main__':
    # save()
    load()
