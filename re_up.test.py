import time
import datetime
import random
from  tello_sdk_stand import *
from robomaster import led
import sys
import threading
import cv2

"""先过矮门，再过高门，从左边绕旗杆"""



dj = Start()



try:
    """起飞"""
    dj.led_obj.set_mled_sc()  # 关闭LED灯，节约用电
    dj.take_off()  # 起飞
    time.sleep(1)
    # print('第一次开始')
    # dj.go(x=100, y=100, z=30, speed=30)
    # print('第一次飞行完成' )
    # # time.sleep(2)
    # # dj.flight_obj.stop()
    # print('+'*40)
    # print('第二次开始')
    # print('='*50)
    # dj.go(x=100, y=-100, z=20, speed=30)
    # print('第二次结束')
    # # time.sleep(3)
    # print("-" * 50)
    # print('第3次开始')
    # dj.go(x=-100, y=-100, z=20, speed=30)
    # print("-" * 50)
    # print('第4次开始')
    # # time.sleep(3)
    # dj.go(x=-100, y=100, z=20, speed=30)
    # print("-" * 50)
    # print('第4次结束')

    dj.curve(x1=100, y1=100, z1=20, x2=200, y2=0, z2=40, speed=30)
    print("-" * 50)
    print('第1次结束')
    dj.curve(x1=-100, y1=-100, z1=20, x2=-200, y2=0, z2=40, speed=30)
    print("+" * 50)
    print('第2次结束')
    # dj.down(20)
    # dj.curve(x1=100, y1=150, z1=30, x2=200, y2=-150, z2=60 , speed=15)
    # dj.curve(x1=100, y1=100, z1=30, x2=200, y2=50, z2=60 , speed=15)

    # time.sleep(2)
    # dj.curve(x1=-100, y1=-100, z1=30, x2=-200, y2=0, z2=60 , speed=15)
    # dj.flight_obj.curve(x1=-30, y1=100, z1=60, x2=100, y2=170, z2=120 , speed=10)
    # dj.go_sys(x=150, y=0, z=0,  speed=40)
    # dj.go(x=150, y=0, z=0,  speed=40)
    time.sleep(2)
    # dj.flight_obj.go(x=-100, y=0, z=0,  speed=10)
    # dj.flight_obj.go(x=0, y=0, z=-40,  speed=20)
    # time.sleep(3)
    dj.land()
except Exception as e:
    print('飞行异常准备降落，错误代码：', e)
    dj.land()  # 降落
finally:
    dj.close()  # 释放飞机资源
    sys.exit()