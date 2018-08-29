import os
import json
import Basic
# 获取数据集
# current=os.getcwd()
# print('current',current)
# DATADIR = os.path.abspath(os.path.join(current,os.path.pardir,os.path.pardir,'dataset/data'))
DATADIR="/mnt/Data/weiyumei/deeplearning/dataset/data"
print('DATADIR',DATADIR)

IMG_PATH=DATADIR+'/train'
MARK_PATH=DATADIR+'/marks' 
# 位置标记文件
filedir=DATADIR+'/annotations.json'
# 每张图片名称所关联的id
ids=open(DATADIR+'/test/ids.txt').read().splitlines()
anns=json.loads(open(filedir).read())
# print('anns',anns['types'])
# 为类别生成数字编号
targets=anns['types']
target_classes=dict()
for i,name in enumerate(targets):
    target_classes[name]=(i+1,name)
#     当读取不到数据，为背景图
target_classes['None']=0
#labels type为数字
labels=target_classes
print(target_classes)
