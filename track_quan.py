from tello_sdk_stand import *
from yolov5_new.detect import DetectApi
from yolov5_new.tt_api import get_qi_info,get_quan_info
import time, sys

model = DetectApi(weights=['.\\yolov5_new\\weights\\best.pt'], nosave=False)
print('识别模型加载完毕')
dj = Start()  # 初始化飞机对象
dj.take_off()  # 起飞
dj.up(30)
for i in range(6):
    print(f'第{i}次跟踪')
    try:

        # time.sleep(5)

        # dj.take_photo()
        # time.sleep(2)
        img_path = dj.take_photo(num=2)
        print('图片路径：',img_path)
        qi_info = get_quan_info(model,img_path)
        print(qi_info)
        if qi_info['code'] == 'action':
            dj.fly(direction=qi_info['direction'],distance=qi_info['distance'])
            if not qi_info['finish']:
                img_path =dj.take_photo(num=2)
                qi_info = get_quan_info(model, img_path)
                dj.fly(direction=qi_info['direction'], distance=qi_info['distance'])
                print('需要再次调用拍照定位')
        elif qi_info['code'] == 'no action':
            print('无需调整位置')
        else:
            # code 为err
            if not qi_info['finish']:
                print('未检测到圈，向右 向后飞行后再试')
                dj.back(40)
                dj.left(40)
                img_path = dj.take_photo(num=2)
                print('图片路径：', img_path)
                qi_info = get_quan_info(model, img_path)
                print(qi_info)
                if qi_info['code'] == 'action':
                    dj.fly(direction=qi_info['direction'], distance=qi_info['distance'])
                    if not qi_info['finish']:
                        img_path = dj.take_photo(num=2)
                        qi_info = get_quan_info(model, img_path)
                        dj.fly(direction=qi_info['direction'], distance=qi_info['distance'])
            else:
                print('代码有bug:',qi_info['code'])

        dj.take_photo(num=1)
        # time.sleep(1)
        # dj.take_photo()

        i+=1


    except Exception as e:
        print('飞行异常准备降落，错误代码：', e)
        dj.land()  # 降落

print('循环完毕，降落')
dj.land()  # 降落
dj.close()