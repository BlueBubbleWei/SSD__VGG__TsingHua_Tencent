import tensorflow as tf
import pascalvoc_to_tfrecords
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'dataset_name', 'TsingHua_Tencent',
    'The name of the dataset to convert.')
# 数据集的路径
tf.app.flags.DEFINE_string(
    'dataset_dir', r'/mnt/Data/weiyumei/deeplearning/dataset/data/train',
    'Directory where the original dataset is stored.')
tf.app.flags.DEFINE_string(
    'output_name', 'TsingHua_Tencent_train',
    'Basename used for TFRecords output files.')
tf.app.flags.DEFINE_string(
    'output_dir', r'/mnt/Data/weiyumei/deeplearning/dataset/data/Tfrecords',
    'Output directory where to store TFRecords files.')


def main(_):
    # print('FLAGS.dataset_dir',FLAGS.dataset_dir)

    if not FLAGS.dataset_dir:
        raise ValueError('You must supply the dataset directory with --dataset_dir')
    print('Dataset directory:', FLAGS.dataset_dir)
    print('Output directory:', FLAGS.output_dir)

    if FLAGS.dataset_name == 'TsingHua_Tencent':
        pascalvoc_to_tfrecords.run(FLAGS.dataset_dir, FLAGS.output_dir, FLAGS.output_name)
    else:
        raise ValueError('Dataset [%s] was not recognized.' % FLAGS.dataset_name)

if __name__ == '__main__':
    tf.app.run()
