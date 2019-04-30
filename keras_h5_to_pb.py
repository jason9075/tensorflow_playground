import cv2
import keras
import numpy as np
import tensorflow as tf
from keras import backend as K
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import node_def_pb2
from keras.models import Sequential
from utils import utils
from keras.layers import Input, Dense, Reshape
from keras.models import Model

MODEL_PATH = 'output_models/age_0506_1.h5'


def main():
    face_vector = np.array([
        -0.09448673, 0.00299672, -0.01515014, -0.00071408, -0.06696047,
        -0.0049022, -0.08832493, -0.08441716, 0.10042, -0.12468411, 0.20260036,
        -0.04197515, -0.14975297, -0.07462036, -0.06249582, 0.18168075,
        -0.17225666, -0.08203805, -0.00451902, 0.02619861, 0.11311005,
        -0.00396955, -0.03501457, 0.06844106, -0.13450187, -0.33966419,
        -0.1363975, -0.07410653, 0.01615451, -0.04746429, -0.03059909,
        -0.00604952, -0.13447005, 0.00071517, 0.02433086, 0.08668517,
        0.00277232, -0.05964718, 0.17318156, 0.04651684, -0.20049778,
        0.05507299, 0.02470755, 0.23595715, 0.19114207, 0.10148107, 0.03782246,
        -0.12871203, 0.09433473, -0.18688148, 0.02312719, 0.11872759,
        0.06432167, 0.08553082, -0.03720454, -0.11119911, 0.0646906,
        0.06421642, -0.14366747, 0.03704356, 0.04679065, -0.07703813,
        0.03313119, 0.01389773, 0.23284599, 0.05472896, -0.12956738,
        -0.16850038, 0.11573525, -0.18238996, -0.09238973, 0.01605953,
        -0.1146273, -0.20592251, -0.31073043, 0.04182778, 0.44439369,
        0.16161928, -0.19643666, 0.07449412, -0.06447967, -0.03075972,
        0.17345226, 0.16626933, 0.02761606, -0.05479746, -0.10204716,
        0.00131448, 0.24683532, -0.05268429, -0.01487479, 0.20475397,
        -0.00205846, 0.0964495, 0.01042209, 0.06929132, -0.09276011,
        0.08609562, -0.01308689, -0.0279386, 0.10474412, 0.01610085,
        0.07669854, 0.12831081, -0.15205586, 0.21715966, -0.03564852,
        0.02630121, 0.08200177, 0.04180234, -0.13742532, -0.02357174,
        0.13010965, -0.21079631, 0.21230085, 0.17257966, 0.0704184, 0.12778719,
        0.15773121, 0.12556361, -0.01399942, 0.00308351, -0.2362863,
        -0.02304427, 0.03029038, -0.02955271, 0.11292487, 0.01218744
    ], dtype=np.float32)

    model = keras.models.load_model(MODEL_PATH)

    inputs = Input(shape=(128, 1))
    x = Reshape((128,))(inputs)
    output2 = model(x)
    model2 = Model(inputs=inputs, outputs=output2)
    model2.summary()

    print('input node name: ', [node.op.name for node in model2.inputs])
    print('output node name: ', [node.op.name for node in model2.outputs])

    with K.get_session() as sess:
        # frozen graph
        additional_nodes = ['input_1', 'sequential_1/dense_6/Relu']
        frozen_graph = utils.freeze_session(sess, output_names=additional_nodes)
        frozen_graph = strip(frozen_graph, 'sequential_1/dropout_3', 'sequential_1/dense_4/Elu',
                             'sequential_1/dense_5/MatMul', 'sequential_1/training')
        frozen_graph = strip(frozen_graph, 'sequential_1/dropout_4', 'sequential_1/dense_5/Elu',
                             'sequential_1/dense_6/MatMul', 'sequential_1/training')
        tf.summary.FileWriter("tensor_board/", graph=frozen_graph)

        input_tensor = sess.graph.get_tensor_by_name('input_1:0')

        # save model to pb file
        tf.train.write_graph(frozen_graph, "output_models/", "age_0506_1.pb", as_text=False)
        tf.train.write_graph(frozen_graph, "output_models/", "age_0506_1.pbtxt", as_text=True)

        # print('keras: ', model2.predict(np.reshape(face_vector, (1, 128,1))))

        print('Inference:')

        net = cv2.dnn.readNetFromTensorflow('output_models/tf_frozen_fashion.pb')
        blob = cv2.dnn.blobFromImage(np.reshape(face_vector, (128, 1, 1)))
        net.setInput(blob)
        detection = net.forward()
        print('opencv: ', detection)


def strip(input_graph, drop_scope, input_before, output_after, pl_name):
    input_nodes = input_graph.node
    nodes_after_strip = []
    for node in input_nodes:
        # print("{0} : {1} ( {2} )".format(node.name, node.op, node.input))

        if node.name.startswith(drop_scope + '/'):
            # print('drop {}'.format(node.name))
            continue

        if node.name.startswith(pl_name + '/'):
            # print('drop {}'.format(node.name))
            continue

        new_node = node_def_pb2.NodeDef()
        new_node.CopyFrom(node)
        if new_node.name == output_after:
            new_input = []
            for node_name in new_node.input:
                if node_name == drop_scope + '/cond/Merge':
                    new_input.append(input_before)
                else:
                    new_input.append(node_name)
            del new_node.input[:]
            new_node.input.extend(new_input)
        nodes_after_strip.append(new_node)

    output_graph = graph_pb2.GraphDef()
    output_graph.node.extend(nodes_after_strip)
    return output_graph


if __name__ == '__main__':
    main()
