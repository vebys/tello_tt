from tello_sdk_stand import *
from yolov5_new.detect import DetectApi
from yolov5_new.tt_api import get_qi_info,get_quan_info
import time, sys

def fly_test():
    model = DetectApi(weights=['.\\yolov5_new\\weights\\best.pt'], nosave=False)
    print('识别模型加载完毕')
    dj = Start()  # 初始化飞机对象
    dj.take_off()  # 起飞
    dj.up(30)
    time.sleep(3)
    focal_list = []
    for i in range(10):
        time.sleep(2)
        print(f'第{i}次计算')
        img_path = dj.take_photo(num=2)
        print('图片路径：', img_path)
        qi_info = get_quan_info(model, img_path)
        if isinstance(qi_info,dict):
            continue
        focal_list.append(qi_info)
    print(focal_list)
    sorted(focal_list)
    focal_list.pop(0)
    focal_list.pop(-1)
    print(focal_list)
    s=0
    for a in focal_list:
        s +=a
    avg_focal = s/len(focal_list)
    print('平均焦距：',avg_focal)

    dj.land()
    dj.close()

def get_avg():
    li = [830.26,862.5,969,847,1036,928,1000,929,1042]
    li.pop(0)
    li.pop(-1)
    sum = 0

    for l in li:
        sum+=l
    avg = sum/len(li)
    print('avg:',avg)

if __name__ == "__main__":
    get_avg()