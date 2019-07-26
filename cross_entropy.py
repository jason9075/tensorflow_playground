import tensorflow as tf


def main():
    tf.enable_eager_execution()

    labels = [0, 3]
    logit = [[4.5, 4.5, 4.5, 4.5],
             [4.5, 4.5, 4.5, 4.5]]


    # result = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=labels)
    result = tf.realdiv(logit,0.5)

    print(result)


if __name__ == '__main__':
    main()
