import pathlib

import tensorflow as tf
from google.protobuf import text_format
from tensorflow.python.platform import gfile

PB_FILE = "output_models/astra_person_detector.pb"


def load_pb_file():
    graph = tf.Graph()
    with graph.as_default():
        graph_def = tf.GraphDef()
        with gfile.FastGFile(PB_FILE, 'rb') as f:
            if pathlib.Path(PB_FILE).suffix == '.pbtxt':
                text_format.Parse(f.read(), graph_def)
            else:
                graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name="pb")
    tf.summary.FileWriter("tensor_board/", graph=graph)


def build_graph():
    graph = tf.Graph()
    with graph.as_default() as graph:
        n = 100
        x = tf.constant(list(range(n)))
        c = lambda i, x: i < n
        b = lambda i, x: (tf.Print(i + 1, [i]), tf.Print(x + 1, [i], "x:"))
        i, out = tf.while_loop(c, b, (0, x))
        with tf.Session() as sess:
            print(sess.run(i))  # prints [0] ... [9999]
            print(sess.run(out).shape)

    tf.summary.FileWriter("tensor_board/", graph=graph)
    tf.train.write_graph(graph, "output_models/", "while.pb", as_text=False)


def main():
    load_pb_file()
    # build_graph()


if __name__ == '__main__':
    main()
