import tensorflow as tf
from tensorflow.python.platform import gfile

PB_FILE = "output_models/astra_person_detector.pb"


def main():
    graph = tf.Graph()
    with graph.as_default():
        graph_def = tf.GraphDef()
        with gfile.FastGFile(PB_FILE, 'rb') as f:
            graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name="pb")
    tf.summary.FileWriter("tensor_board/", graph=graph)


if __name__ == '__main__':
    main()
