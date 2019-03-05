import tensorflow as tf


def main():
    with tf.Graph().as_default() as graph:
        a = tf.constant(1, name="a")
        b = tf.constant(2, name="b")
        c = tf.add(a, b, name="c")
        split = tf.split([0, 1, 2, 3, 4], 5, name="split_op")  # return list of tensor
        sum = tf.add(split[3], c, name="sum")

    tf.summary.FileWriter("tensor_board/", graph=graph)

    tensor_0 = graph.get_tensor_by_name('split_op:0')
    tensor_1 = graph.get_tensor_by_name('split_op:1')
    tensor_2 = graph.get_tensor_by_name('split_op:2')
    tensor_3 = graph.get_tensor_by_name('split_op:3')
    tensor_4 = graph.get_tensor_by_name('split_op:4')

    sess = tf.Session(graph=graph)

    print('by name_0: {}, by parameter:{}'.format(sess.run(tensor_0), sess.run(split[0])))
    print('by name_1: {}, by parameter:{}'.format(sess.run(tensor_1), sess.run(split[1])))
    print('by name_2: {}, by parameter:{}'.format(sess.run(tensor_2), sess.run(split[2])))
    print('by name_3: {}, by parameter:{}'.format(sess.run(tensor_3), sess.run(split[3])))
    print('by name_4: {}, by parameter:{}'.format(sess.run(tensor_4), sess.run(split[4])))
    print('sum: {}'.format(sess.run(sum)))

    sess.close()


if __name__ == '__main__':
    main()
