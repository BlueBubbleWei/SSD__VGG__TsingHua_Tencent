import os
import json
import tensorflow as tf
import numpy as np
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
        'image/shape': tf.FixedLenFeature([3], tf.int64),
        'image/object/bbox/xmin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/xmax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/label': tf.VarLenFeature(dtype=tf.int64),
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
        labels_to_names[pair[0]] = name
    # print('label_names',labels_to_names)
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

    # [image, shape, glabels_raw, gbboxes_raw] = provider.get(['image', 'shape',
    #                                                                    'object/label','object/bbox'])


    with tf.Session() as sess:
        sess.run([tf.local_variables_initializer(), tf.global_variables_initializer()])
        tf.train.start_queue_runners()
        for i in range(provider._num_samples):
            [image, labelList, boxList, shape] = provider.get(
                ['image', 'object/label', 'object/bbox', 'shape'])
            img, labels, boxes, shape = sess.run([image, labelList, boxList, shape])
            print(labels)

            # img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)#不转换反而是RGB显示
            # print('{}is ,has shape :{}'.format(img, shape))
            # img=cv2.imread(img)
            # img = img / 255.0#归一化以后会化成黑色
            for j in range(len(labels)):
                print('value:', ( boxes[j][0], boxes[j][1]), ( boxes[j][2], boxes[j][3]))
                cv2.rectangle(img, (int(boxes[j][0] * shape[0]), int(boxes[j][1] * shape[1])),
                              (int(boxes[j][2] * shape[0]), int(boxes[j][3] * shape[1])), (0, 255, 0), 3)
            plt.imshow(img)
            plt.show()



            cv2.imwrite("./rec.jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            # plt.show()

            break


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
    slim_get_batch(223, 1, 'train',os.path.join(dataset_dir, os.path.pardir, 'dealTsinghua_Tecent/Tsinghua_Tecent_000.tfrecord'), 100, 4, )


if __name__ == '__main__':
    main()
