from tello_sdk_stand import *
from yolov5_new.detect import DetectApi
from yolov5_new.tt_api import get_qi_info,get_quan_info
import time, sys

model = DetectApi(weights=['.\\yolov5_new\\weights\\best.pt'], nosave=False)
print('识别模型加载完毕')
dj = Start()  # 初始化飞机对象
dj.take_off()  # 起飞
dj.up(30)
focal_list = []
for i in range(20):
    print(f'第{i}次跟踪')
    img_path = dj.take_photo(num=2)
    print('图片路径：', img_path)
    qi_info = get_quan_info(model, img_path)
    focal_list.append(qi_info)
print(focal_list)
sorted(focal_list)
focal_list.pop(0)
focal_list.pop(-1)
print(focal_list)
a=0
for a in focal_list:
    a +=a
avg_focal = a/len(focal_list)
print('平均焦距：',avg_focal)