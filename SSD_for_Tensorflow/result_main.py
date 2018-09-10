"""
date: 2017/11/10
author: lslcode [jasonli8848@qq.com]
"""

import os
import gc
import random
import skimage.io
import skimage.transform
import numpy as np
import tensorflow as tf
import ssd300
import json
import glob
import anno_func
import cv2
'''
SSD检测
'''
def testing():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        ssd_model = ssd300.SSD300(sess,False)
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(var_list=tf.trainable_variables())
        if os.path.exists('./session_params/session.ckpt.index') :
            saver.restore(sess, './session_params/session.ckpt')
            image, actual,file_list = get_traindata_voc2007(1)
            pred_class, pred_class_val, pred_location = ssd_model.run(image,None)

            print('file_list:' + str(file_list))
            
            for index, act in zip(range(len(image)), actual):
                for a in act :
                    print('【img-'+str(index)+' actual】:' + str(a))
                print('pred_class:' + str(pred_class[index]))
                print('pred_class_val:' + str(pred_class_val[index]))
                print('pred_location:' + str(pred_location[index]))   
                                   
        else:
            print('No Data Exists!')
        sess.close()
    
'''
SSD训练
'''
def training():
    batch_size = 8
    running_count = 0
    
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        ssd_model = ssd300.SSD300(sess,True)
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(var_list=tf.trainable_variables())
        if os.path.exists('./session_params/session.ckpt.index') :
            print('\nStart Restore')
            saver.restore(sess, './session_params/session.ckpt')
            print('\nEnd Restore')
         
        print('\nStart Training')
        min_loss_location = 100000.
        min_loss_class = 100000.
        while((min_loss_location + min_loss_class) > 0.001 and running_count < 100000):
            running_count += 1
            
            train_data, actual_data,_ = get_traindata_voc2007(batch_size)
            if len(train_data) > 0:
                loss_all,loss_class,loss_location,pred_class,pred_location = ssd_model.run(train_data, actual_data)
                l = np.sum(loss_location)
                c = np.sum(loss_class)
                if min_loss_location > l:
                    min_loss_location = l
                if min_loss_class > c:
                    min_loss_class = c
               
                print('Running:【' + str(running_count) + '】|Loss All:【'+str(min_loss_location + min_loss_class)+'|'+ str(loss_all) + '】|Location:【'+ str(np.sum(loss_location)) + '】|Class:【'+ str(np.sum(loss_class)) + '】|pred_class:【'+ str(np.sum(pred_class))+'|'+str(np.amax(pred_class))+'|'+ str(np.min(pred_class)) + '】|pred_location:【'+ str(np.sum(pred_location))+'|'+str(np.amax(pred_location))+'|'+ str(np.min(pred_location)) + '】')
                
                # 定期保存ckpt
                if running_count % 100 == 0:
                    saver.save(sess, './session_params/session.ckpt')
                    print('session.ckpt has been saved.')
                    gc.collect()
            else:
                print('No Data Exists!')
                break
            
        saver.save(sess, './session_params/session.ckpt')
        sess.close()
        gc.collect()
            
    print('End Training')
    
'''
获取voc2007训练图片数据
train_data：训练批次图像，格式[None,width,height,3]
actual_data：图像标注数据，格式[None,[None,center_x,center_y,width,height,lable]]
'''
# file_name_list = os.listdir('../../dataset/VOC2007/JPEGImages/')
# lable_arr = ['background','aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']
dataset_dir=os.path.join(os.path.curdir,os.path.pardir,os.path.pardir,'/dataset/data/')
ANNOTATIONS= os.path.join(dataset_dir,'annotations.json')
anns=json.loads(open(ANNOTATIONS).read())
CLASSES=anns['types']
classes=dict()
for i,name in enumerate(CLASSES):
    classes[name]=(i+1,name)
classes['None']=(0,'Background')
ids=os.path.join(dataset_dir,'ids.txt')
file_name_list = glob.glob(dataset_dir+'train/*.jpg')
lable_arr=[ name[1] for name in classes.values()]
# 图像白化，格式:[R,G,B]
whitened_RGB_mean = [123.68/255, 116.78/255, 103.94/255]
def get_traindata_voc2007(batch_size):
    def splitImg(imgpath):
        image = skimage.io.imread(img_path)
        image = image[400:1424, :, :]
        split_list = np.hsplit(image, 2)
        img_list = []
        for img in split_list:
            img = skimage.transform.resize(img, (300, 300))
            img = img - whitened_RGB_mean
            img_list.append(img)
        return img_list

    def get_actual_data_from_xml(img_path):
        actual_item = []
        img2 = []
        img1 = []
        try:
            img_id = img_path.split('\\')[-1].split('.')[0]
            # datadir = os.path.join(dataset_dir, os.path.pardir)
            imgdata, img_msg = anno_func.load_img(anns, dataset_dir, img_id)
            imgdata = imgdata[400:1424, :, :]
            split_list = np.hsplit(imgdata, 2)
            split_interval = len(split_list[0])
            shape = split_list[0].shape
            img_width, img_height, depth = shape
            img_info = img_msg['objects']

            for info in img_info:
                label = info['category'].strip()
                label = classes[label][0]
                xmin = info['bbox']['xmin']
                xmax = info['bbox']['ymin']
                ymin = info['bbox']['xmax'] - 400.0
                ymax = info['bbox']['ymax'] - 400.0
                # 1024.0为设定的剪切图片的大小
                if ymin > 1024.0:
                    break
                if ymax > 1024.0:
                    ymax = 1024.0
                if xmin < split_interval and xmax < split_interval:
                    xmin, xmax = xmin, xmax
                    img1.append([((xmin + xmax) / 2 / img_width), ((ymin + ymax) / 2 / img_height),
                                 ((xmin - xmax) / img_width),
                                 ((ymin - ymax) / img_height), label])
                elif xmin < split_interval and xmax < 2 * split_interval:
                    xmin, xmax = xmin, split_interval
                    img1.append([((xmin + xmax) / 2 / img_width), ((ymin + ymax) / 2 / img_height),
                                 ((xmin - xmax) / img_width),
                                 ((ymin - ymax) / img_height), label])
                elif (xmin >= split_interval and xmin < 2 * split_interval) and (
                        xmax < 2 * split_interval):  # 超越第一张图片的大小
                    xmin, xmax = xmin - split_interval, xmax - split_interval
                    img2.append([((xmin + xmax) / 2 / img_width), ((ymin + ymax) / 2 / img_height),
                                 ((xmin - xmax) / img_width),
                                 ((ymin - ymax) / img_height), label])
                elif (xmin >= split_interval and xmin < 2 * split_interval) and (xmax >= 2 * split_interval):  # 一倍间距
                    xmin, xmax = xmin - split_interval, xmax - split_interval
                    img2.append([((xmin + xmax) / 2 / img_width), ((ymin + ymax) / 2 / img_height),
                                 ((xmin - xmax) / img_width),
                                 ((ymin - ymax) / img_height), label])
                else:
                    continue
            actual_item.extend([img1, img2])
            return actual_item
        except:
            return None
    train_data = []
    actual_data = []
    
    file_list = random.sample(file_name_list, batch_size)
    for f_name in file_list:
        img_path = f_name
        if os.path.splitext(img_path)[1].lower() == '.jpg':
            # 分割成两张图片
            img_list = splitImg(img_path)
            #两张图片对应的标签
            actual_items = get_actual_data_from_xml(img_path)
            for index, actual_item in enumerate(actual_items):
                #如果标签为空那么图片也不要了
                if actual_item != None and len(actual_item) != 0:
                    actual_data.append(actual_item)
                    train_data.append(img_list[index])
    return train_data, actual_data,file_list


def main():
    training()
'''
主程序入口
'''
if __name__ == '__main__':
    print('\nStart Running')
    # 检测
    #testing()
    # 训练
    main()
    print('\nEnd Running')
   
        
