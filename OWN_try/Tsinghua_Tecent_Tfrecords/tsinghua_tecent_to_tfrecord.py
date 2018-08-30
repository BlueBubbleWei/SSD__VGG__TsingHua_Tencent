import os
import random
import json
import glob
import  sys
import tensorflow as tf
import anno_func
from dataset_utils import int64_feature,float_feature,bytes_feature
RANDOM_SEED = 4242
SAMPLES_PER_FILES = 100

def get_annotations(dataset_dir):
    '''获取数据集的标注信息'''
    ANNOTATIONS= os.path.join(dataset_dir,os.path.pardir,'annotations.json')
    anns=json.loads(open(ANNOTATIONS).read())
    CLASSES=anns['type']
    classes=dict()
    for i,name in enumerate(CLASSES):
        classes[name]=(i+1,name)
    classes['None']=(0,'Background')
    ids=os.path.join(dataset_dir,'ids.txt')
    return CLASSES

def process_image(dataset_dir,filename):
    '''获取每张图片的信息'''
    filename_id = filename.split('\\')[-1].split('.')[0]
    ANNOTATIONS = os.path.join(dataset_dir, os.path.pardir, 'annotations.json')
    ids = open(dataset_dir + 'ids.txt').read().splitlines()
    annotations = json.loads(open(ANNOTATIONS).read())
    CLASSES = annotations['types']
    classes = dict()
    for i, name in enumerate(CLASSES):
        classes[name] = (i + 1, i+1)
    classes['Background'] = (0, 0)
    # filename_id='571'
    dataset_dir=os.path.join(dataset_dir,os.path.pardir)#为了配合annos存储的是'path': 'test/33132.jpg'
    image_data = tf.gfile.FastGFile(filename, 'rb').read()

    imgdata,img = anno_func.load_img(annotations, dataset_dir, filename_id)
    shape=imgdata.shape
    img_info=img['objects']
    bboxes = []
    labels = []
    labels_text = []
    xmin=[]
    ymin=[]
    xmax=[]
    ymax=[]
    for info in img_info:
        labels_text.append(info['category'].encode('utf-8'))
        labels.append(classes[info['category']][0])
        xmin.append(info['bbox']['xmin'])
        ymin.append(info['bbox']['ymin'])
        xmax.append(info['bbox']['xmax'])
        ymax.append(info['bbox']['ymax'])
        bbox = info['bbox']
        bboxes.append([bbox['xmin']/shape[0], bbox['ymin']/shape[1], bbox['xmax']/shape[0], bbox['ymax']/shape[1]])
        # r_e_shape.append(info)
    return image_data,shape,labels,labels_text,bboxes


def convert_to_example(imgdata, shape, labels, labels_text, bboxes):
    '''转换数据'''
    xmin = []
    ymin = []
    xmax = []
    ymax = []
    for b in bboxes:
        assert len(b) == 4
        [l.append(point) for l, point in zip([ymin, xmin, ymax, xmax], b)]

    image_format=b'JPEG'
    example=tf.train.Example(features=tf.train.Features(feature={
        'image/height': int64_feature(shape[0]),
        'image/width': int64_feature(shape[1]),
        'image/channel': int64_feature(shape[2]),
        'image/shape': int64_feature(list(shape)),
        'image/object/bbox/xmin': float_feature(xmin),
        'image/object/bbox/xmax': float_feature(xmax),
        'image/object/bbox/ymin': float_feature(ymin),
        'image/object/bbox/ymax': float_feature(ymax),
        'image/object/bbox/label': int64_feature(labels),
        'image/object/bbox/label_text': bytes_feature(labels_text),
        'image/format': bytes_feature(image_format),
        'image/encoded': bytes_feature(imgdata)
    }))
    return example



def add_to_record(dataset_dir,filename,tfrecorder_writer):
    imgdata, shape, labels, labels_text, bboxes=process_image(dataset_dir,filename)
    example=convert_to_example(imgdata, shape, labels, labels_text, bboxes)
    tfrecorder_writer.write(example.SerializeToString())




def run(dataset_dir,dataset_name,output_dir):
    #是否打乱数据
    shuffling = False
    #数据集是否存在
    if not tf.gfile.Exists(dataset_dir):
        raise ValueError('数据集不存在！')
    else:
        # 获取所有的图片列表'/dataset/data/train\\10020.jpg'
        images=glob.glob(dataset_dir+'*.jpg')

        # classes=get_annotations(dataset_dir)
    i=0
    fidx=0
    while i < len(images):
        print('输出名称列表')
        # 输出文件的名称
        tf_filename = '%s/%s_%03d.tfrecord' % (output_dir, dataset_name, fidx)
        print('tf_filename',tf_filename)
        with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
            j=0
            while i < len(images) and j < SAMPLES_PER_FILES:
                sys.stdout.write('Converting %d/%d images'%(i+1,len(images)))
                sys.stdout.flush()
                filename=images[i]
                # filename_id=filename.split('/')[-1].split('.')[0]
                add_to_record(dataset_dir,filename,tfrecord_writer)
                i+=1
                j+=1
            fidx+=1
    print('\n Finished converting Tsinghua_Tecent dataset.')

    if shuffling:
        random.random(RANDOM_SEED)
    pass