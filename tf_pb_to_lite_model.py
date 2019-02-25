import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from tensorflow.python.platform import gfile

from utils.mnist_reader import load_mnist

LESS_DATA = True


def softmax(x):
    score_mat_exp = np.exp(np.asarray(x))
    return score_mat_exp / score_mat_exp.sum(0)


def main():
    x_test, y_test = load_mnist('data/fashion', kind='t10k')

    if LESS_DATA:
        x_test, y_test = x_test[0:1000, ], y_test[0:1000, ]

    graph = tf.Graph()
    with graph.as_default():
        graph_def = tf.GraphDef()
        with gfile.FastGFile("output_models/tf_frozen_fashion.pb", 'rb') as f:
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name="pb")

    input_tensor = graph.get_tensor_by_name('pb/input:0')
    output_tensor = graph.get_tensor_by_name('pb/output_dense/BiasAdd:0')

    with tf.Session(graph=graph) as sess:
        output_result = sess.run(
            output_tensor, feed_dict={input_tensor: x_test})
        output_logit = np.argmax(output_result, axis=1)

        print("pb file result: {:.2f}".format(
            accuracy_score(y_test, output_logit)))

    # Convert to lite

    # converter = tf.lite.TFLiteConverter.from_frozen_graph("output_models/tf_frozen_fashion.pb", ['input'], ['output_dense/BiasAdd'])
    # tflite_model = converter.convert()
    #
    # with open("output_models/tf_frozen_fashion.tflite", "wb") as f:
    #     f.write(tflite_model)
    #
    # converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
    # tflite_model_size = converter.convert()
    # with open("output_models/tf_frozen_fashion_size.tflite", "wb") as f:
    #     f.write(tflite_model_size)
    #
    # converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_LATENCY]
    # tflite_model_latency = converter.convert()
    # with open("output_models/tf_frozen_fashion_latency.tflite", "wb") as f:
    #     f.write(tflite_model_latency)

    # eval lite performance:

    interpreter = tf.lite.Interpreter(model_path="output_models/tf_frozen_fashion.tflite")
    interpreter.allocate_tensors()
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]

    output_result = []
    for idx, y in enumerate(y_test):
        interpreter.set_tensor(input_index, x_test.astype(np.float32)[idx:idx + 1])
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_index)
        output_result.append(np.argmax(prediction))

    print("tf_frozen_fashion result: {:.2f}".format(
        accuracy_score(y_test, output_result)))


if __name__ == '__main__':
    # 請用command 執行 使用pycharm 會失敗
    main()
