from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import cv2
slim = tf.contrib.slim

data_splits_num = {
    'train': 6105
}
dataset_dir=os.path.join(os.path.curdir,os.path.pardir,os.path.pardir,'/dataset/data/')
ANNOTATIONS= os.path.join(dataset_dir,'annotations.json')
anns=json.loads(open(ANNOTATIONS).read())
CLASSES=anns['types']
classes=dict()
for i,name in enumerate(CLASSES):
    classes[name]=(i+1,name)
classes['None']=(0,'Background')
ids=os.path.join(dataset_dir,'ids.txt')

def slim_get_batch(num_classes, batch_size, split_name, file_pattern, num_readers, num_epochs=None, is_training=True):
    """获取一个数据集元组，其中包含有关读取P数据集的说明。
    Args:
      num_classes:数据集中的总类数。
      batch_size: the size of each batch.
      split_name: 'train' of 'val'.
      file_pattern: 匹配数据集源时使用的文件模式（完整路径）。
      num_readers: 用于阅读tfrecords的最大阅读器数量。
      num_preprocessing_threads: 用于运行预处理功能的最大线程数。
      image_preprocessing_fn: 用于数据集扩充的函数。
      anchor_encoder: 用于编码所有锚点的函数。
      num_epochs: 用于迭代此数据集的总epoches。
      is_training:
    Returns:     allow_smaller_final_batch=(not is_training),
                    num_threads=num_preprocessing_threads,
                    capacity=64 * batch_size)
    """
    if split_name not in data_splits_num:
        raise ValueError('split name %s was not recognized.' % split_name)

    # Features in Pascal VOC TFRecords.
    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='jpeg'),
        'image/height': tf.FixedLenFeature([1], tf.int64),
        'image/width': tf.FixedLenFeature([1], tf.int64),
        'image/channels': tf.FixedLenFeature([1], tf.int64),
        'image/shape': tf.FixedLenFeature([3], tf.int64),
        'image/object/bbox/xmin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/xmax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/label': tf.VarLenFeature(dtype=tf.int64),
        'image/object/bbox/label_text': tf.VarLenFeature(dtype=tf.int64),
    }
    items_to_handlers = {
        'image': slim.tfexample_decoder.Image('image/encoded', 'image/format'),
        'shape': slim.tfexample_decoder.Tensor('image/shape'),
        'object/bbox': slim.tfexample_decoder.BoundingBox(
            ['ymin', 'xmin', 'ymax', 'xmax'], 'image/object/bbox/'),
        'object/label': slim.tfexample_decoder.Tensor('image/object/bbox/label'),
    }
    decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)

    labels_to_names = {}
    for name, pair in classes.items():
        labels_to_names[name] = name

    dataset = slim.dataset.Dataset(
        data_sources=file_pattern,
        reader=tf.TFRecordReader,
        decoder=decoder,
        num_samples=data_splits_num[split_name],
        items_to_descriptions=None,
        num_classes=num_classes,
        labels_to_names=labels_to_names)

    with tf.name_scope('dataset_data_provider'):
        provider = slim.dataset_data_provider.DatasetDataProvider(
            dataset,
            num_readers=num_readers,
            common_queue_capacity=32 * batch_size,
            common_queue_min=8 * batch_size,
            shuffle=is_training,
            num_epochs=num_epochs)

    # [image, shape, glabels_raw, gbboxes_raw] = provider.get(['image','shape',
    #                                                                    'object/label',
    #                                                                    'object/bbox'])

    with tf.Session() as sess:
        sess.run([tf.local_variables_initializer(), tf.global_variables_initializer()])
        tf.train.start_queue_runners()
        for i in range(provider._num_samples):
            [image, labelList, boxList, shape] = provider.get(
                ['image', 'object/label', 'object/bbox', 'shape'])
            # image = tf.image.decode_jpeg(image)
            image = tf.decode_raw(image, tf.int64)
            image, labels, boxes, shape = sess.run([image, labelList, boxList, shape])
            print(labelList)
            # print('{}is ,{} has shape :{}'.format(image, filename.decode('utf-8'), shape))
            # print('{}is ,{} has shape :{}'.format(image, filename.decode('utf-8'), shape))
            for j in range(labels.shape[0]):
                print('label=%d (%s): locations in %.6f, %.6f, %.6f, %.6f' % (
                labels[j], labels_to_names[labels[j]], boxes[j][0], boxes[j][1], boxes[j][2], boxes[j][3]))
            break
    return image, labels, boxes, shape

def count_split_examples(split_path, file_prefix='.tfrecord'):
    # Count the total number of examples in all of these shard
    num_samples = 0
    tfrecords_to_count = tf.gfile.Glob(os.path.join(split_path, file_prefix))
    opts = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)
    for tfrecord_file in tfrecords_to_count:
        for record in tf.python_io.tf_record_iterator(tfrecord_file):#, options = opts):
            num_samples += 1
    return num_samples



def main():
    datadir_tfrecords=os.path.join(dataset_dir ,os.path.pardir,'dealTsinghua_Tecent/Tsinghua_Tecent_000.tfrecord')
    print(tf.gfile.Exists(datadir_tfrecords))

    filename_queue = tf.train.string_input_producer([datadir_tfrecords])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)  # 返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'image/shape': tf.FixedLenFeature([], tf.int64),
                                           'image/encoded': tf.FixedLenFeature([], tf.string),
                                       })

    # img = tf.decode_raw(features['image/encoded'], tf.uint8)
    # img = tf.reshape(img, [224, 224, 3])
    # img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    shape = features[ 'image/shape']
    print(shape)
    img_batch, label_batch = tf.train.shuffle_batch([ shape],
                                                    batch_size=4, capacity=2000,
                                                    min_after_dequeue=1000)
    init = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init)
        threads = tf.train.start_queue_runners(sess=sess)
        for i in range(3):
            val, l = sess.run([img_batch, label_batch])
            # 我们也可以根据需要对val， l进行处理
            # l = to_categorical(l, 12)
            print(val.shape, l)

    # image, labels, boxes, shape=slim_get_batch(223,4,'train',datadir_tfrecords,100,4,)
    # for serialized_example in tf.python_io.tf_record_iterator(datadir_tfrecords):
    #     example = tf.train.Example()
    #     example.ParseFromString(serialized_example)
    #
    #     shape = example.features.feature['image/shape'].int64_list.value
    #     image = example.features.feature['image/encoded'].bytes_list.value
    #     image = tf.decode_raw(image, tf.uint8)
    #     label = example.features.feature['image/object/bbox/label'].int64_list.value
    #     # 可以做一些预处理之类的
    #     print(shape, image,label)


if __name__ == '__main__':
    import numpy as np
    img_raw = np.random.randint(0, 255, size=(56, 56))
    print(img_raw)
    img_raw = img_raw.tostring()
    # print(img_raw)
    img_raw = tf.gfile.FastGFile(os.path.join(os.path.curdir,os.path.pardir,os.path.pardir,'/dataset/data/train/36.jpg'), 'rb').read()
    # print(img_raw)
     # main()