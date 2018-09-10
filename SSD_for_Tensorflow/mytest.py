import os
import glob
import json
import anno_func
import cv2
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import skimage.io
import skimage.transform
import tensorflow as tf
sns.set_style("darkgrid",{"font.sans-serif":['KaiTi', 'Arial']})

# 数据集的基本信息


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



# 获取最大最小y的极限值

"""
_xmin=0.0
_ymin=0.0
_xmax=0.0
_ymax=0.0
tops=[]
count500=0
in_interval=0
print('图片数量为',len(anns['imgs']))
for obj in anns['imgs']:
    objs=anns['imgs'][obj]
    if ('train') in objs['path']:
        objs=objs['objects']
        for value in range(len(objs)):
            obj=objs[value]
            xmin=obj['bbox']['xmin']
            xmax = obj['bbox']['xmax']
            if xmin< _xmin:_xmin=xmin
            if xmax > _xmax: _xmax = xmax
            ymin=obj['bbox']['ymin']
            ymax = obj['bbox']['ymax']
            tops.append(ymin)
            tops.append(ymax)
            if ymin<300:count500+=1
            if ymax > _ymax: _ymax = ymax
            if xmin<1024.0 and xmax>1024.0:in_interval+=1
print("横跨两张图片的box的数量为:",in_interval)
print('小于500一下的像素为',count500)
print('len(tops)',len(tops))
# plt.bar(range(len(tops)), tops)
# plt.ylabel(r"交通标志在图片中的高度")
# plt.xlabel(r"各图片中交通标志高度方向上的位置总数")
# plt.show()
print(_xmin,_xmax,_ymin,_ymax)
"""

# 高度上边界框的位置分布
"""
name_list = ['高度<100','高度<200','高度<300','高度<400','高度<500']
yboxs=[28,102,174,339,647]
plt.bar(range(len(yboxs)),yboxs,tick_label=name_list)
plt.ylabel(r"交通标志高度方向上的位置框的坐标分布")
plt.show()
"""

# 获取图片信息
"""
img_id='23'
imgdata, img_msg=anno_func.load_img(anns, dataset_dir, img_id)
"""

# 显示标记的图形
"""
img_id='1404'
imgdata, img_msg= anno_func.load_img(anns, dataset_dir,img_id )
mask=anno_func.load_mask(anns, dataset_dir, img_id, imgdata)
imgs=anno_func.draw_all(anns, dataset_dir,  img_id, imgdata)
# imgs=imgs[500:1500,600:1600,:]
# mask=mask[500:1500,600:1600]
#RGB->BGR
imgs=cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)
imgs=imgs[...,[2,1,0]]#为什么非要这么转换
cv2.imshow('img1',imgs)
cv2.imshow('img2',mask)
cv2.waitKey()
cv2.destroyAllWindows()
"""

# cv2.destroyAllWindows()
#SSD图片的位置坐标是中心点坐标加width,hright,[centerX,centerY,width,height]

img_id="307"
img_path=os.path.join(os.path.curdir,os.path.pardir,os.path.pardir,'/dataset/data/train/'+img_id+'.jpg')
print(img_path)

#经测量从高度为400到1424的地方截取，将图片分割成两张1024*1024大小的图片
"""
img=cv2.imread(img_path)
img2=img[400:1424,:,:]
split_list=np.hsplit(img2,2)
#图片的框框在两张图片中的分布
imgdata, img_msg= anno_func.load_img(anns, dataset_dir,img_id )
img_info=img_msg['objects']
split_interval=len(split_list[0])
boxes_xy=dict()
img1=[]
img2=[]
for info in range(len(img_info)):
    boxes=img_info[info]['bbox']
    xmin=boxes['xmin']
    xmax=boxes['xmax']
    ymin=boxes['ymin']-400.0
    ymax=boxes['ymax']-400.0
    # 1024.0为设定的剪切图片的大小
    if ymin>1024.0:
        break
    if ymax > 1024.0:
        ymax=1024.0
    if xmin<split_interval and xmax<split_interval:
        xmin,xmax=xmin,xmax
        img1.append((int(xmin), int(ymin), int(xmax), int(ymax)))
    elif xmin<split_interval and xmax< 2*split_interval:
        xmin, xmax = xmin, split_interval
        img1.append((int(xmin), int(ymin), int(xmax), int(ymax)))
    elif (xmin >= split_interval and xmin < 2* split_interval)  and (xmax < 2*split_interval):#超越第一张图片的大小
        xmin, xmax = xmin-split_interval, xmax-split_interval
        img2.append((int(xmin), int(ymin), int(xmax), int(ymax)))
    elif (xmin >= split_interval and xmin < 2 * split_interval) and (xmax >= 2 * split_interval):  # 一倍间距
        xmin, xmax = xmin - split_interval, xmax - split_interval
        img2.append((int(xmin), int(ymin), int(xmax), int(ymax)))
    else:break
boxes_xy['img1']=img1
boxes_xy['img2'] = img2
print(boxes_xy.values())

print('split_list[0]',len(split_list[0]),split_list[0].shape)
for  i in range(len(split_list)):
    boxes=list(boxes_xy.values())[i]
    for box in boxes:
        cv2.rectangle(split_list[i],(box[0],box[1]),(box[2],box[3]) , (0, 255, 0), 2)
    cv2.imshow(str(i),split_list[i])
cv2.waitKey()
cv2.destroyAllWindows()
"""

# 分割成两张图片并获取相应的图片
"""
def get_actual_data_from_xml(img_path):
    actual_item = []
    img2=[]
    img1=[]
    try:
        img_id = img_path.split('/')[-1].split('.')[0]
        # datadir = os.path.join(dataset_dir, os.path.pardir)
        imgdata, img_msg = anno_func.load_img(anns, dataset_dir, img_id)
        imgdata = imgdata[400:1424, :, :]
        split_list = np.hsplit(imgdata, 2)
        img_info = img_msg['objects']
        split_interval = len(split_list[0])
        shape = split_list[0].shape
        img_width, img_height, depth = shape
        img_info = img_msg['objects']

        for info in img_info:
            label = info['category'].strip()
            label = classes[label][0]
            xmin = info['bbox']['xmin']
            xmax = info['bbox']['ymin']
            ymin = info['bbox']['xmax']-400.0
            ymax = info['bbox']['ymax']-400.0
            # 1024.0为设定的剪切图片的大小
            if ymin > 1024.0:
                break
            if ymax > 1024.0:
                ymax = 1024.0
            if xmin < split_interval and xmax < split_interval:
                xmin, xmax = xmin, xmax
                img1.append( [((xmin + xmax) / 2 / img_width), ((ymin + ymax) / 2 / img_height),
                 ((xmin - xmax) / img_width),
                 ((ymin - ymax) / img_height), label])
            elif xmin < split_interval and xmax < 2 * split_interval:
                xmin, xmax = xmin, split_interval
                img1.append( [((xmin + xmax) / 2 / img_width), ((ymin + ymax) / 2 / img_height),
                 ((xmin - xmax) / img_width),
                 ((ymin - ymax) / img_height), label])
            elif (xmin >= split_interval and xmin < 2 * split_interval) and (xmax < 2 * split_interval):  # 超越第一张图片的大小
                xmin, xmax = xmin - split_interval, xmax - split_interval
                img2.append( [((xmin + xmax) / 2 / img_width), ((ymin + ymax) / 2 / img_height),
                 ((xmin - xmax) / img_width),
                 ((ymin - ymax) / img_height), label])
            elif (xmin >= split_interval and xmin < 2 * split_interval) and (xmax >= 2 * split_interval):  # 一倍间距
                xmin, xmax = xmin - split_interval, xmax - split_interval
                img2.append( [((xmin + xmax) / 2 / img_width), ((ymin + ymax) / 2 / img_height),
                 ((xmin - xmax) / img_width),
                 ((ymin - ymax) / img_height), label])
            else:
                continue
        actual_item.extend([img1,img2])
        print(actual_item)
        return actual_item
    except:
        return None

def splitImg(imgpath):
    image=skimage.io.imread(img_path)
    image = image[400:1424, :, :]
    split_list = np.hsplit(image, 2)
    img_list=[]
    for img in split_list:
        img = skimage.transform.resize(img, (300, 300))
        img = img - whitened_RGB_mean
        img_list.append(img)
    return img_list
actual_data=[]
train_data=[]
whitened_RGB_mean = [123.68, 116.78, 103.94]
if os.path.splitext(img_path)[1].lower() == '.jpg':
    img_list=splitImg(img_path)
    actual_items = get_actual_data_from_xml(img_path)
    for index,actual_item in enumerate(actual_items):
        if actual_item != None and len(actual_item)!=0:
            print('actual_item',actual_item)
            actual_data.append(actual_item)
            train_data.append(img_list[index])
        else:
            print('Error : ' + img_path)
"""



