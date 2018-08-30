import os
import tensorflow as tf
FLAGS=tf.app.flags.FLAGS
import tsinghua_tecent_to_tfrecord

#定义变量
tf.app.flags.DEFINE_string('dataset_dir',os.path.join(os.path.curdir,os.path.pardir,os.path.pardir,'/dataset/data/train/'),'数据集的位置')
tf.app.flags.DEFINE_string('dataset_name','Tsinghua_Tecent','数据集的名字')
tf.app.flags.DEFINE_string('output_dir',os.path.join(os.path.curdir,os.path.pardir,os.path.pardir,'/dataset/dealTsinghua_Tecent/'),'输出文件的目录')

def main():
    tsinghua_tecent_to_tfrecord.run(FLAGS.dataset_dir,FLAGS.dataset_name,FLAGS.output_dir)
if __name__ == '__main__':
    main()
