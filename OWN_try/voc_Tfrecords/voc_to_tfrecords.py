import os
import random
import sys
import xml.etree.ElementTree as ET
import voc_common
from dataset_utils import int64_feature,float_feature,bytes_feature
import tensorflow as tf
# 数据集下的路径
DIRECTORY_ANNOTATIONS = 'Annotations/'
DIRECTORY_IMAGES = 'JPEGImages/'

RANDOM_SEED = 4242
SAMPLES_PER_FILES = 200

def process_image(dataset_dir,imgname):
    '''根据图片的名称获取每张图片的信息'''
    filename=dataset_dir+DIRECTORY_IMAGES+ imgname+'.jpg'
    print('\n filename',filename)
    #读取图片
    image_data=tf.gfile.FastGFile(filename,'rb').read()
    filename=os.path.join(dataset_dir,DIRECTORY_ANNOTATIONS,imgname+'.xml')
    # 解析xml文件
    tree=ET.parse(filename)
    root=tree.getroot()

    # 获取图片信息
    size=root.find('size')
    shape=[int(size.find('height').text),
           int(size.find('width').text),
           int(size.find('depth').text)
           ]
    bboxes=[]
    labels=[]
    labels_text=[]
    difficult=[]
    truncated=[]

    for obj in root.findall('object'):
        label=obj.find('name').text
        labels.append(int(voc_common.VOC_LABELS[label][0]))
        labels_text.append(label.encode('ascii'))

        if obj.find('difficult'):
            difficult.append(int(obj.find('difficult').text))
        else:
            difficult.append(0)
        if obj.find('truncated'):
            truncated.append(int(obj.find('truncated').text))

        bbox=obj.find('bndbox')
        bboxes.append((
            float(bbox.find('ymin').text)/shape[0],
            float(bbox.find('xmin').text)/shape[1],
            float(bbox.find('ymax').text)/shape[0],
            float(bbox.find('ymin').text)/shape[1]
        ))
    return image_data,shape,bboxes,labels,labels_text,difficult,truncated

def convert_to_example(image_data,labels,labels_text,bboxes,shape,difficult,truncated):
    '''转换每一张图片'''
    xmin=[]
    ymin=[]
    xmax=[]
    ymax=[]
    for b in bboxes:
        assert len(b) == 4
        [l.append(point) for l,point in zip([ymin,xmin,ymax,xmax],b)]

    image_format=b'JPEG'
    example=tf.train.Example(features=tf.train.Features(feature={
        'image/height':int64_feature(shape[0]),
        'image/width':int64_feature(shape[1]),
        'image/channel':int64_feature(shape[2]),
        'image/shape':int64_feature(shape),
        'image/object/bbox/xmin':float_feature(xmin),
        'image/object/bbox/xmax':float_feature(xmax),
        'image/object/bbox/ymin':float_feature(ymin),
        'image/object/bbox/ymax':float_feature(ymax),
        'image/object/bbox/label':int64_feature(labels),
        'image/object/bbox/label_text':bytes_feature(labels_text),
        'image/object/bbox/difficult':int64_feature(difficult),
        'image/object/bbox/truncated':int64_feature(truncated),
        'image/format':bytes_feature(image_format),
        'image/encoded':bytes_feature(image_data)
    }))
    return example


def add_to_tfrecord(dataset_dir,file_name,tfrecord_writer):
    image_data,shape,bboxes,labels,labels_text,difficult,truncated=process_image(dataset_dir,file_name)
    example=convert_to_example(image_data,labels,labels_text,bboxes,shape,difficult,truncated)
    tfrecord_writer.write(example.SerializeToString())

def run(dataset_dir,dataset_name,output_dir):
    shuffling=False
    if not tf.gfile.Exists(dataset_dir):
        raise ValueError('数据集的路径不存在！')
    path=os.path.join(dataset_dir,DIRECTORY_ANNOTATIONS)
    filenames=sorted(os.listdir(path))
    if shuffling:
        random.seed(RANDOM_SEED)
        random.shuffle(filenames)

    i=0
    fidx=0
    while i< len(filenames):
        print('输出名字列表')
        #输出文件的名称
        tf_filename='%s/%s_%03d.tfrecord' % (output_dir, dataset_name, fidx)
        # 写入文件
        print('tf_filename',tf_filename)
        tf_filename=os.path.join(tf_filename)
        print('tf_filename', tf_filename)
        # open(tf_filename,'wb').read()
        with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
            j=0
            # SAMPLES_PER_FILES是指每个tfrecord文件保存的图片的数量,每更新一轮就开始一个新的record
            while i < len(filenames) and j < SAMPLES_PER_FILES:
                sys.stdout.write('\r>>Converting images %d/%d' %(i+1,len(filenames)))
                sys.stdout.flush()

                filename=filenames[i]
                img_name=filename[:-4]
                add_to_tfrecord(dataset_dir,img_name,tfrecord_writer)
                i += 1
                j += 1
            fidx +=1
    print('\n Finished converting VOC dataset.')