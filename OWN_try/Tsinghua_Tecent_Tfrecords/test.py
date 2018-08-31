import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

writer = tf.python_io.TFRecordWriter('test.tfrecord')

for i in range(0, 2):
    a = np.random.random(size=(180)).astype(np.float32)
    a = a.data.tolist()
    b = [2016 + i, 2017 + i]
    c = np.array([[0, 1, 2], [3, 4, 5]]) + i
    c = c.astype(np.uint8)
    c = tf.gfile.FastGFile(os.path.join(os.path.curdir, os.path.pardir, os.path.pardir, '/dataset/data/train/36.jpg'), 'rb').read()

    # c_raw = tf.image.decode_jpeg(c)  # 图像解码

    # img_raw = tf.image.convert_image_dtype(img_raw, dtype=tf.uint8)  # 改变图像数据类型
    # with tf.Session() as sess:
    #     imgage=sess.run(img_raw)
    #     print('imgage',type(img_raw))
    # c_raw = c.tostring()  # 这里是把ｃ换了一种格式存储
    # print('  i:', i)
    # print('  a:', a)
    # print('  b:', b)
    # print('  c:', c)
    # print('c_raw:',c_raw)
    # c=tf.decode_raw(c)
    example = tf.train.Example(features=tf.train.Features(
        feature={'a': tf.train.Feature(float_list=tf.train.FloatList(value=a)),
                 'b': tf.train.Feature(int64_list=tf.train.Int64List(value=b)),
                 'c': tf.train.Feature(bytes_list=tf.train.BytesList(value=[c]))}))
    serialized = example.SerializeToString()
    writer.write(serialized)
    # print('   writer', i, 'DOWN!')
writer.close()

filename_queue = tf.train.string_input_producer(['test.tfrecord'], num_epochs=None)
# create a reader from file queue
reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)
# get feature from serialized example

features = tf.parse_single_example(serialized_example,
                                   features={
                                       'a': tf.FixedLenFeature([180], tf.float32),
                                       'b': tf.FixedLenFeature([2], tf.int64),
                                       'c': tf.FixedLenFeature([], tf.string)
                                   }
                                   )
a_out = features['a']
b_out = features['b']
c_out = features['c']
# c_raw_out = features['c']
# c_raw_out = tf.sparse_to_dense(features['c'])
# c_out = tf.decode_raw(c_raw_out, tf.uint8)
# print(a_out)
# print(b_out)
# print(c_out)
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     print(sess.run(c_out))
a_batch, b_batch, c_batch = tf.train.shuffle_batch([a_out, b_out, c_out], batch_size=3,
                                                   capacity=200, min_after_dequeue=100, num_threads=2)


with tf.Session() as sess:  # 开始一个会话
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    try:
        # while not coord.should_stop():
        for i in range(2):
            example, cimage = sess.run([b_batch, c_batch])  # 在会话中取出image和label
            print(type(cimage),cimage.dtype)


            cimage = tf.image.decode_jpeg(cimage)  # 图像解码
            print(sess.run(cimage))  # 打印解码后的图像（即为一个三维矩阵[w,h,3]）
            cimage = tf.image.convert_image_dtype(cimage, dtype=tf.uint8)  # 改变图像数据类型
            # img = tf.image.convert_image_dtype(cimage, dtype=tf.uint8)  # 改变图像数据类型
            # print('img',example   )
            # img=np.array(img, dtype=float)
            # cimage=np.array(cimage,dtype=np.float32)
            plt.imshow(cimage)
            plt.show()
    except tf.errors.OutOfRangeError:
        print('Done reading')
    finally:
        coord.request_stop()

    coord.request_stop()
    coord.join(threads)








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


def transform():
    # img_raw = np.random.randint(0, 255, size=(56, 56))
    # img_raw = img_raw.tostring()
    tfrecords_filename='test.tfrecord'
    tfrecords_filename=os.path.join(os.path.curdir,tfrecords_filename)
    writer = tf.python_io.TFRecordWriter(tfrecords_filename)  # 创建.tfrecord文件，准备写入
    # print(tf.gfile.Exists(os.path.join(os.path.curdir,os.path.pardir,os.path.pardir,'/dataset/data/train/36.jpg')))
    img_raw = tf.gfile.FastGFile(os.path.join(os.path.curdir,os.path.pardir,os.path.pardir,'/dataset/data/train/36.jpg'), 'rb').read()



    # print(sess.run(image_jpg))  # 打印解码后的图像（即为一个三维矩阵[w,h,3]）
    # with tf.Session() as sess:
    #     image_jp=sess.run(image_jpg)
    #

    # img_raw = img_raw.tostring()
    example = tf.train.Example(features=tf.train.Features(
        feature={
            'image/shape': tf.train.Feature(int64_list=tf.train.Int64List(value=[2048,2048,3])),
            'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
        }))
    writer.write(example.SerializeToString())
    writer.close()

def main():
    datadir_tfrecords=os.path.join(dataset_dir ,os.path.pardir,'dealTsinghua_Tecent/Tsinghua_Tecent_000.tfrecord')
    transform()
    filename_queue = tf.train.string_input_producer(['test.tfrecord'])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)  # 返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'image/shape': tf.FixedLenFeature([], tf.int64),
                                           'image/encoded': tf.FixedLenFeature([], tf.string),
                                           'image/object/bbox/xmin': tf.FixedLenFeature([], tf.float32),
                                           'image/object/bbox/xmax': tf.FixedLenFeature([], tf.float32),
                                           'image/object/bbox/ymin': tf.FixedLenFeature([], tf.float32),
                                           'image/object/bbox/ymax': tf.FixedLenFeature([], tf.float32),
                                       })

    img = features['image/encoded']
    image_jpg = tf.image.decode_jpeg(img)  # 图像解码
    with tf.Session() as sess:
        imgs=sess.run(image_jpg)
        img = tf.image.convert_image_dtype(imgs, dtype=tf.uint8)  # 改变图像数据类型
        plt.imshow(img)
        plt.show()
    shape = features[ 'image/shape']
    # xmin=features[ 'image/object/bbox/xmin']
    # xmax = features['image/object/bbox/xmax']
    # ymin = features['image/object/bbox/ymin']
    # ymax = features['image/object/bbox/ymax']
    batch_size = 3
    min_after_dequeue = 10
    capacity = min_after_dequeue + 3 * batch_size
    # , shape, xmin, xmax, ymin, ymax
    image,shape = tf.train.shuffle_batch([img,shape],
                                          batch_size=batch_size,
                                          num_threads=3,
                                          capacity=200,
                                          min_after_dequeue=100)

    with tf.Session() as sess:  # 开始一个会话
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        try:
            # while not coord.should_stop():
            for i in range(2):
                example,s= sess.run([image,shape])  # 在会话中取出image和label
                plt.imshow(example)
                plt.show()
        except tf.errors.OutOfRangeError:
            print('Done reading')
        finally:
            coord.request_stop()

        coord.request_stop()
        coord.join(threads)



if __name__ == '__main__':
    main()
