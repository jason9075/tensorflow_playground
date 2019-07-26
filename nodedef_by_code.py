import tensorflow as tf


def main():
    input_tensor = tf.placeholder(
        name='input_images',
        shape=[5],
        dtype=tf.float32)

    x = tf.add(input_tensor, 5)

    result_tensor = tf.multiply(x, 10)

    with tf.Session() as sess:
        result = sess.run(result_tensor, feed_dict={input_tensor: [1, 2, 3, 4, 5]})
        print(result)
        tf.io.write_graph(sess.graph.as_graph_def(), "output_models/", "node_def.pb", as_text=True)


if __name__ == '__main__':
    main()
