from img.tello_sdk_stand import *

dj = Start()

try:
    """起飞"""
    # dj.led_obj.set_mled_sc()  # 关闭LED灯，节约用电
    dj.take_off()  # 起飞
    time.sleep(1)
    # 螺旋上升
    dj.curve(x1=85, y1=35, z1=20, x2=170, y2=0, z2=30, speed=30)
    dj.curve(x1=-85, y1=-35, z1=20, x2=-170, y2=20, z2=30, speed=30)

    # 环绕山峰
    # 飞到山峰前
    dj.forward(200)
    # 调整左右位置
    dj.right(30)
    # dj.take_photo('before',num=4)
    # 下降高度
    dj.down(70)
    # 定位到距离悬崖60-80厘米的位置  ，智慧调整前后位置，不会调整左右
    cliff_loc(dj)
    # 悬停3秒
    time.sleep(3)
    # dj.take_photo('ing',num=4)

    # 旋转360°
    dj.reverse(360)
    # dj.take_photo('after',num=4)


    # 飞跃悬崖
    # 上升
    dj.up(70)
    # 向前飞
    dj.forward(290)

    # 特级表演
    # 下降高度
    dj.down(90)
    # 后退
    dj.back(120)
    # 上升
    dj.up(90)
    # 向前
    dj.forward(130)
    # 右飞
    dj.right(100)
    # 降落
    dj.land()
except Exception as e:
    print('飞行异常准备降落，错误代码：', e)
    dj.land()  # 降落

dj.close()  # 释放飞机资源



