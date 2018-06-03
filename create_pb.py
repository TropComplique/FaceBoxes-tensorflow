import tensorflow as tf
import argparse

"""Create a .pb frozen inference graph from a SavedModel."""


def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-s', '--saved_model_folder', type=str
    )
    parser.add_argument(
        '-o', '--output_pb', type=str, default='model.pb'
    )
    return parser.parse_args()


def main():

    graph = tf.Graph()
    config = tf.ConfigProto()
    config.gpu_options.visible_device_list = '0'
    with graph.as_default():
        with tf.Session(graph=graph, config=config) as sess:
            tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], ARGS.saved_model_folder)

            # output ops
            keep_nodes = ['boxes', 'scores', 'num_boxes']

            input_graph_def = tf.graph_util.convert_variables_to_constants(
                sess, graph.as_graph_def(),
                output_node_names=keep_nodes
            )
            output_graph_def = tf.graph_util.remove_training_nodes(
                input_graph_def,
                protected_nodes=keep_nodes + [n.name for n in input_graph_def.node if 'nms' in n.name]
            )
            # ops in 'nms' scope must be protected for some reason,
            # but why?

            with tf.gfile.GFile(ARGS.output_pb, 'wb') as f:
                f.write(output_graph_def.SerializeToString())
            print('%d ops in the final graph.' % len(output_graph_def.node))


ARGS = make_args()
tf.logging.set_verbosity('INFO')
main()
