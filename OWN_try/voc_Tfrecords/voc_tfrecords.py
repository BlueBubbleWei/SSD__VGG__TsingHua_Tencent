import os
import tensorflow as tf
import voc_to_tfrecords

FLAGS=tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('dataset_name','voc_2007_train',"dataset'name'")
tf.app.flags.DEFINE_string('dataset_dir',os.path.join(os.path.curdir, os.path.pardir, os.path.pardir, '/dataset/VOC2007/'),'数据集的路径')
tf.app.flags.DEFINE_string('output_dir',os.path.join(os.path.curdir, os.path.pardir, os.path.pardir,'/dataset/dealVOC/'),'Tfrecord的输出路径')

def main():
    print(os.listdir(FLAGS.output_dir),FLAGS.output_dir)
    voc_to_tfrecords.run(FLAGS.dataset_dir,FLAGS.dataset_name,FLAGS.output_dir)


if __name__ == '__main__':
    main()