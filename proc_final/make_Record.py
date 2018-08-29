from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os
import random
import sys
import threading
import numpy as np
import six
import tensorflow as tf
import json
import glob
import anno_func
import Basic


def get_num(img_path):
    '''根据图片的名字截取后面的数字'''
    return str(img_path.split('/')[-1].split('.')[0])


def get_img_info(img):
    '''读取单张图片的信息'''
    imgid = img.split('.')[0]
    imgdata = anno_func.load_img(anns, DATADIR, imgid)
    imgdata_draw, mask_ellipse, img = anno_func.draw_all(anns, DATADIR, imgid, imgdata)
    img_info = img['objects']
    label_classes = []
    label_text = []
    bbox_list = []
    ymin = []
    xmin = []
    ymax = []
    xmax = []
    for obj in range(len(img_info)):
        label_text.append(img_info[obj]['category'].encode())
        bbox = img_info[obj]['bbox']
        assert len(bbox) == 4
        bbox_list.append([bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']])
        label_classes.append(int(target_classes[img_info[obj]['category']][0]))
    return bbox_list, label_classes, label_text


RANDOM_SEED = 180428


def _int64_feature(value):
    """将int64类型加入到实例原型的装饰器"""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
    """将float类型加入到实例原型的装饰器"""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_list_feature(value):
    """将二进制列表类型加入到实例原型的装饰器"""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _bytes_feature(value):
    """将二进制类型加入到实例原型的装饰器"""
    if isinstance(value, six.string_types):
        value = six.binary_type(value, encoding='utf-8')
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _convert_to_example(filename, image_name, image_buffer, bboxes, labels, labels_text,
                        height, width):
    """
    构建一个示例proto作为示例。
    args：
     filename：string，图像文件的路径，例如'/path/to/example.JPG'
     image_buffer：字符串，RGB图像的JPEG编码
     bboxes：每个图像的边界框列表
     labels：边界框的标签列表
     labels_text：边界框的标签名称列表
     difficult：整数列表表明该边界框的难度
     truncated：整数列表表示该边界框的截断
     height：整数，图像高度（以像素为单位）
     width：整数，图像宽度（以像素为单位）
     Returns：
     Example proto    
    """
    ymin = []
    xmin = []
    ymax = []
    xmax = []
    for b in bboxes:
        assert len(b) == 4
        # pylint: disable=expression-not-assigned
        [l.append(point) for l, point in zip([ymin, xmin, ymax, xmax], b)]
    channels = 3
    image_format = b'JPEG'
    image_data = tf.gfile.FastGFile(filename, 'rb').read()
    #     循环获取信息，加入list
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': _int64_feature(height),
        'image/width': _int64_feature(width),
        'image/channels': _int64_feature(channels),
        'image/shape': _int64_feature([height, width, channels]),
        'image/object/bbox/xmin': _float_feature(xmin/width),
        'image/object/bbox/xmax': _float_feature(xmax/width),
        'image/object/bbox/ymin': _float_feature(ymin/height),
        'image/object/bbox/ymax': _float_feature(ymax/height),
        'image/object/bbox/label': _int64_feature(labels),
        'image/object/bbox/label_text': _bytes_list_feature(labels_text),
        'image/format': _bytes_feature(image_format),
        'image/filename': _bytes_feature(filename.encode('utf8')),
        'image/encoded': _bytes_feature(image_buffer)}))
    return example


class ImageCoder(object):
    """Helper class that provides TensorFlow image coding utilities."""

    def __init__(self):
        # sess 回调所有的图片
        self._sess = tf.Session()

        # 将png转成jpg.
        self._png_data = tf.placeholder(dtype=tf.string)
        image = tf.image.decode_png(self._png_data, channels=3)
        self._png_to_jpeg = tf.image.encode_jpeg(image, format='rgb', quality=100)

        # 将CMYK JPEG 转成 RGB JPEG d
        self._cmyk_data = tf.placeholder(dtype=tf.string)
        image = tf.image.decode_jpeg(self._cmyk_data, channels=0)
        self._cmyk_to_rgb = tf.image.encode_jpeg(image, format='rgb', quality=100)

        # 解码RGB
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

    def png_to_jpeg(self, image_data):
        return self._sess.run(self._png_to_jpeg,
                              feed_dict={self._png_data: image_data})

    def cmyk_to_rgb(self, image_data):
        return self._sess.run(self._cmyk_to_rgb,
                              feed_dict={self._cmyk_data: image_data})

    def decode_jpeg(self, image_data):
        image = self._sess.run(self._decode_jpeg,
                               feed_dict={self._decode_jpeg_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image


def _process_image(filename, coder):
    """处理单张图片
    Args:
    filename: '/path/to/example.JPG'.
    coder: instance of ImageCoder to provide TensorFlow image coding utils.
    Returns:
    image_buffer: string, JPEG encoding of RGB image.
    height: integer, image height in pixels.
    width: integer, image width in pixels.
    """
    # 读取图片.
    with tf.gfile.FastGFile(filename, 'rb') as f:
        image_data = f.read()

    # 解码RGB JPEG.
    image = coder.decode_jpeg(image_data)

    # 转成 RGB
    assert len(image.shape) == 3
    height = image.shape[0]
    width = image.shape[1]
    #     print('image.shape[2]:',image.shape[2])
    assert image.shape[2] == 3

    return image_data, height, width


def _process_image_files_batch(coder, thread_index, ranges, name, directory, all_records, num_shards):
    """在单线程中处理并保存为 TFRecord 图片列表
    Args:
    coder: ImageCoder实例提供TensorFlow图像编码工具。
    thread_index: integer, unique batch to run index is within [0, len(ranges)).
    ranges: 指定要并行分析的每个批次范围的整数对列表。
    name: string,指定数据集的唯一标识符
    directory: string; 所有数据的路径
    all_records: 字符串元组列表; 每个元组的第一个是记录的子目录，第二个是图像文件名。
    num_shards: num_shards：此数据集的整数分片数。
    """
    # Each thread produces N shards where N = int(num_shards / num_threads).
    # For instance, if num_shards = 128, and the num_threads = 2, then the first
    # thread would produce shards [0, 64).
    num_threads = len(ranges)
    assert not num_shards % num_threads
    num_shards_per_batch = int(num_shards / num_threads)

    shard_ranges = np.linspace(ranges[thread_index][0],
                               ranges[thread_index][1],
                               num_shards_per_batch + 1).astype(int)
    num_files_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

    counter = 0
    for s in range(num_shards_per_batch):
        # 生成文件名的分片版本, e.g. 'train-00002-of-00010'
        shard = thread_index * num_shards_per_batch + s
        output_filename = '%s_%s_%s_%03d.tfrecord' % (name, test, shard,num_shards)
        # output_filename = '%s-%.5d-of-%.5d.tfrecord' % (name, shard, num_shards)
        output_file = os.path.join(output_directory, output_filename)
        # output_file = output_filename
        with open(output_file, 'wb'):
            writer = tf.python_io.TFRecordWriter(output_file)

            shard_counter = 0
            files_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
            #     print('files_in_shard',files_in_shard)
            for i in files_in_shard:
                cur_record = all_records[i]
                filename = os.path.join(directory, cur_record[0], cur_record[1])

                image_buffer, height, width = _process_image(filename, coder)
                bboxes, labels, labels_text = get_img_info(cur_record[1])
                example = _convert_to_example(filename, cur_record[1], image_buffer, bboxes, labels, labels_text,
                                              height, width)
                writer.write(example.SerializeToString())
                #             print(height,width,shard_counter,example)
                shard_counter += 1
                counter += 1

                if not counter % 1000:
                    print('%s [thread %d]: Processed %d of %d images in thread batch.' %
                          (datetime.now(), thread_index, counter, num_files_in_thread))
                sys.stdout.flush()

            writer.close()
        print('%s [thread %d]: Wrote %d images to %s' %
              (datetime.now(), thread_index, shard_counter, output_file))
        sys.stdout.flush()
        shard_counter = 0
    print('%s [thread %d]: Wrote %d images to %d shards.' %
          (datetime.now(), thread_index, counter, num_files_in_thread))
    sys.stdout.flush()


def _process_image_files(name, directory, all_records, num_threads, num_shards):
    """处理并保存图像列表作为示例protos的TFRecord。
    Args:
    name: string, unique identifier specifying the data set
    directory: string; the path of all datas
    all_records:字符串元组列表; 每个元组的第一个是记录的子目录，第二个是图像文件名。 
    num_shards: integer number of shards for this data set.
    """
    # Break all images into batches with a [ranges[i][0], ranges[i][1]].
    spacing = np.linspace(0, len(all_records), num_threads + 1).astype(np.int)
    ranges = []
    threads = []
    for i in range(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i + 1]])

    # 为每个批次启动一个线程。
    print('Launching %d threads for spacings: %s' % (num_threads, ranges))
    sys.stdout.flush()

    # 创建一个监视所有线程何时完成的机制。
    coord = tf.train.Coordinator()

    # 创建一个基于TensorFlow的通用实用程序，用于转换所有图像编码。
    coder = ImageCoder()

    threads = []
    for thread_index in range(len(ranges)):
        args = (coder, thread_index, ranges, name, directory, all_records, num_shards)
        t = threading.Thread(target=_process_image_files_batch, args=args)
        t.start()
        threads.append(t)

    # Wait for all the threads to terminate.
    coord.join(threads)
    print('%s: Finished writing all %d images in data set.' %
          (datetime.now(), len(all_records)))
    sys.stdout.flush()


def _process_dataset(name, directory, all_splits, num_threads, num_shards):
    """处理完整的数据集并将其另存为TFRecord。
    Args:
    name: string, unique identifier specifying the data set.
    directory: string, root path to the data set.
    all_splits: list of strings, sub-path to the data set.
    num_shards: integer number of shards for this data set.
    """
    all_records = []
    for split in all_splits:
        jpeg_file_path = os.path.join(directory, split)
        images = tf.gfile.ListDirectory(jpeg_file_path)
        jpegs = [im_name for im_name in images if im_name.strip()[-3:] == 'jpg']
        all_records.extend(list(zip([split] * len(jpegs), jpegs)))

    shuffled_index = list(range(len(all_records)))
    random.seed(RANDOM_SEED)
    random.shuffle(shuffled_index)
    all_records = [all_records[i] for i in shuffled_index]
    _process_image_files(name, directory, all_records, num_threads, num_shards)


def parse_comma_list(args):
    return [s.strip() for s in args.split(',')]


def main():
    _process_dataset('TsingHua_Tencent', DATADIR, parse_comma_list(test), num_threads, train_shards)


if __name__ == '__main__':
    # 生成Tfrecords数据集
    num_threads = 4
    train_shards = 8
    DATADIR = Basic.DATADIR
    test = 'train'
    anns = Basic.anns
    output_directory = DATADIR + '/Tfrecords'
    target_classes = Basic.target_classes
    print('target_classes:', target_classes)
    main()
