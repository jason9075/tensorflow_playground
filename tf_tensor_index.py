import tensorflow as tf


def main():
    with tf.Graph().as_default() as graph:
        a = tf.constant(1, name="a")
        b = tf.constant(2, name="b")
        c = tf.add(a, b, name="c")
        split = tf.split([0, 1, 2, 3, 4], 5, name="split_op")  # return list of tensor
        sum_value = tf.add(split[3], c, name="sum")

        x = tf.constant([1, 4])
        y = tf.constant([2, 5])
        z = tf.constant([3, 6])
        tf.stack([x, y, z], axis=0, name="stack_axis0")  # [[1, 4], [2, 5], [3, 6]] (Pack along first dim.)
        tf.stack([x, y, z], axis=1, name="stack_axis1")  # [[1, 2, 3], [4, 5, 6]]

    # tf.summary.FileWriter("tensor_board/", graph=graph)

    tensor_0 = graph.get_tensor_by_name('split_op:0')
    tensor_1 = graph.get_tensor_by_name('split_op:1')
    tensor_2 = graph.get_tensor_by_name('split_op:2')
    tensor_3 = graph.get_tensor_by_name('split_op:3')
    tensor_4 = graph.get_tensor_by_name('split_op:4')

    stack_1 = graph.get_tensor_by_name('stack_axis1:0')

    sess = tf.Session(graph=graph)

    print('by tensor_0: {}, by parameter:{}'.format(sess.run(tensor_0), sess.run(split[0])))
    print('by tensor_1: {}, by parameter:{}'.format(sess.run(tensor_1), sess.run(split[1])))
    print('by tensor_2: {}, by parameter:{}'.format(sess.run(tensor_2), sess.run(split[2])))
    print('by tensor_3: {}, by parameter:{}'.format(sess.run(tensor_3), sess.run(split[3])))
    print('by tensor_4: {}, by parameter:{}'.format(sess.run(tensor_4), sess.run(split[4])))
    print('sum: {}'.format(sess.run(sum_value)))
    print('stack_1_0:{}'.format(sess.run(stack_1)))

    sess.close()


if __name__ == '__main__':
    main()
