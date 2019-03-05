import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from tensorflow.python.platform import gfile
import timeit
import cv2

from utils.mnist_reader import load_mnist

LESS_DATA = True
GEN_LITE_MODEL = False
BATCH = 1000
EVAL_LITE_MODEL = "output_models/tf_frozen_fashion.tflite"


def main():
    x_test, y_test = load_mnist('data/fashion', kind='t10k')

    if LESS_DATA:
        x_test, y_test = x_test[0:BATCH, ], y_test[0:BATCH, ]

    # Eval origin pb model
    print("#=== pb model ===#")

    graph = tf.Graph()
    with graph.as_default():
        graph_def = tf.GraphDef()
        with gfile.FastGFile("output_models/tf_frozen_fashion.pb", 'rb') as f:
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name="pb")

    input_tensor = graph.get_tensor_by_name('pb/input:0')
    output_tensor = graph.get_tensor_by_name('pb/output_dense/BiasAdd:0')

    start = timeit.default_timer()
    with tf.Session(graph=graph) as sess:
        output_result = sess.run(
            output_tensor, feed_dict={input_tensor: x_test})
        output_logit = np.argmax(output_result, axis=1)
    end = timeit.default_timer()

    print("pb file acc: {:.2f}".format(
        accuracy_score(y_test, output_logit)))
    print("batch {}'s cost time: {:.3f} sec".format(BATCH, end - start))

    for i in range(0, 5):
        start = timeit.default_timer()
        with tf.Session(graph=graph) as sess:
            prediction = sess.run(
                output_tensor, feed_dict={input_tensor: x_test[i:i + 1]})
        end = timeit.default_timer()
        prediction = np.argmax(prediction)
        print("single cost time: {:.3f} sec, ans:{}, predict:{}".format(end - start, y_test[i], prediction))

    # Convert to lite

    if GEN_LITE_MODEL:
        converter = tf.lite.TFLiteConverter.from_frozen_graph("output_models/tf_frozen_fashion.pb", ['input'],
                                                              ['output_dense/BiasAdd'])
        tflite_model = converter.convert()

        with open("output_models/tf_frozen_fashion.tflite", "wb") as f:
            f.write(tflite_model)

        converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
        tflite_model_size = converter.convert()
        with open("output_models/tf_frozen_fashion_size.tflite", "wb") as f:
            f.write(tflite_model_size)

        converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_LATENCY]
        tflite_model_latency = converter.convert()
        with open("output_models/tf_frozen_fashion_latency.tflite", "wb") as f:
            f.write(tflite_model_latency)

    # Eval lite performance:
    print("#=== tensorflow lite ===#")

    interpreter = tf.lite.Interpreter(model_path=EVAL_LITE_MODEL)
    interpreter.resize_tensor_input(0, [BATCH, 28, 28, 1])
    interpreter.allocate_tensors()
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]

    start = timeit.default_timer()
    interpreter.set_tensor(input_index, x_test.astype(np.float32))
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_index)
    end = timeit.default_timer()

    output_logit = np.argmax(prediction, axis=1)

    print("tf_frozen_fashion acc: {:.2f}".format(
        accuracy_score(y_test, output_logit)))
    print("batch {}'s cost time: {:.3f} sec".format(BATCH, end - start))

    interpreter = tf.lite.Interpreter(model_path=EVAL_LITE_MODEL)
    interpreter.allocate_tensors()

    for i in range(0, 5):
        start = timeit.default_timer()
        interpreter.set_tensor(input_index, x_test.astype(np.float32)[i:i + 1])
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_index)
        end = timeit.default_timer()
        prediction = np.argmax(prediction)
        print("single cost time: {:.3f} sec, ans:{}, predict:{}".format(end - start, y_test[i], prediction))


def gen_model(path, input_node, output_node, output_model_name='gen_model.tflite'):
    # converter = tf.lite.TFLiteConverter.from_frozen_graph(path, [input_node], [output_node])
    converter = tf.lite.TFLiteConverter.from_frozen_graph(path, [input_node], [output_node],
                                                          input_shapes={input_node: [1, 300, 300, 3]})
    converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_LATENCY]
    tflite_model = converter.convert()

    with open("output_models/{}".format(output_model_name), "wb") as f:
        f.write(tflite_model)


def gen_model_from_h5(path, output_model_name='gen_model_h5.tflite'):
    converter = tf.lite.TFLiteConverter.from_keras_model_file(path, input_arrays=['input_1'],
                                                              input_shapes={'input_1': [1, 448, 448, 3]})
    # converter.post_training_quantize = True
    converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_LATENCY]
    tflite_model = converter.convert()

    with open("output_models/{}".format(output_model_name), "wb") as f:
        f.write(tflite_model)


def gen_model_by_input_shape(path, input_node, output_node, shape, output_model_name='gen_model.tflite'):
    graph = tf.Graph()
    with graph.as_default():
        graph_def = tf.GraphDef()
        with gfile.FastGFile(path, 'rb') as f:
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def)

    converter = tf.lite.TFLiteConverter.from_frozen_graph(path, [input_node], [output_node])
    converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_LATENCY]
    tflite_model = converter.convert()

    with open("output_models/{}".format(output_model_name), "wb") as f:
        f.write(tflite_model)


def model_test(pb_model_path, input_node, output_node, lite_model_path):
    # Eval origin pb model
    print("#=== pb model ===#")

    graph = tf.Graph()
    with graph.as_default():
        graph_def = tf.GraphDef()
        with gfile.FastGFile(pb_model_path, 'rb') as f:
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def)

    input_tensor = graph.get_tensor_by_name('{}:0'.format(input_node))
    output_tensor = graph.get_tensor_by_name('{}:0'.format(output_node))

    for i in range(1, 6):
        img = cv2.imread('data/pedestrian/pedestrian{}.jpg'.format(i))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        start = timeit.default_timer()
        with tf.Session(graph=graph) as sess:
            _ = sess.run(
                output_tensor, feed_dict={input_tensor: np.expand_dims(img, axis=0)})
        end = timeit.default_timer()
        print("single cost time: {:.3f} sec".format(end - start))

    # Eval lite performance:
    print("#=== tensorflow lite ===#")

    interpreter = tf.lite.Interpreter(model_path=lite_model_path)
    interpreter.allocate_tensors()
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]

    for i in range(0, 5):
        start = timeit.default_timer()
        interpreter.set_tensor(input_index, x_test.astype(np.float32)[i:i + 1])
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_index)
        end = timeit.default_timer()
        prediction = np.argmax(prediction)
        print("single cost time: {:.3f} sec, ans:{}, predict:{}".format(end - start, y_test[i], prediction))


def h5_model_test(h5_path, lite_path):
    from keras.models import load_model

    model = load_model(h5_path)

    for i in range(1, 4):
        img = cv2.imread('data/kangaroo/k{}.jpg'.format(i))
        img = cv2.resize(img, (448, 448), interpolation=cv2.INTER_CUBIC)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        start = timeit.default_timer()
        model.predict(np.expand_dims(img, axis=0))
        end = timeit.default_timer()
        print("h5 single cost time: {:.3f} sec".format(end - start))

    interpreter = tf.lite.Interpreter(model_path=lite_path)
    interpreter.allocate_tensors()
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]

    for i in range(1, 4):
        img = cv2.imread('data/kangaroo/k{}.jpg'.format(i))
        img = cv2.resize(img, (448, 448), interpolation=cv2.INTER_CUBIC)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = (img/255).astype(np.float32)
        start = timeit.default_timer()
        interpreter.set_tensor(input_index, np.expand_dims(img, axis=0))
        interpreter.invoke()
        interpreter.get_tensor(output_index)
        end = timeit.default_timer()
        print("lite single cost time: {:.3f} sec".format(end - start))


if __name__ == '__main__':
    # 若要GEN_LITE_MODEL 請用command 執行 使用pycharm run 會失敗, 或者參考以下網址 tflite_convert
    # https://www.tensorflow.org/lite/convert/cmdline_examples
    # main()
    # gen_model("output_models/person_vector.pb", 'Placeholder', 'head/emb/BiasAdd')
    # gen_model("output_models/astra_person_detector.pb", 'Preprocessor/mul', 'Postprocessor/BatchMultiClassNonMaxSuppression/map/strided_slice',
    #           'astra_person_detector.tflite')
    gen_model_from_h5('output_models/kangaroo.h5', output_model_name='kangaroo.tflite')
    # model_test("output_models/ssd_mobilenet_v1_coco.pb", 'import/image_tensor', 'import/detection_boxes',
    #            'output_models/ssd_mobilenet_v1_coco.tflite')

    h5_model_test('output_models/kangaroo.h5', 'output_models/kangaroo.tflite')

