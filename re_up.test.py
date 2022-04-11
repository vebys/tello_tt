from img.tello_sdk_stand import *
dj = Start()


try:
    """起飞"""
    # dj.led_obj.set_mled_sc()  # 关闭LED灯，节约用电
    dj.take_off()  # 起飞
    time.sleep(1)
    #螺旋上升
    dj.curve(x1=100, y1=100, z1=20, x2=200, y2=0, z2=40, speed=30)
    dj.curve(x1=-100, y1=-100, z1=20, x2=-180, y2=0, z2=40, speed=30)

    # 环绕山峰
    # 飞到山峰前
    dj.forward(220)
    # 下降高度
    dj.down(90)
    #旋转360°
    dj.reverse(360)


    # 飞跃悬崖
    #上升
    dj.up(90)
    # 向前飞
    dj.forward(230)

    #特级表演
    # 下降高度
    dj.down(90)
    #后退
    dj.back(100)
    #上升
    dj.up(90)
    #向前
    dj.forward(100)
    #右飞
    dj.right(140)
    #降落
    dj.land()
except Exception as e:
    print('飞行异常准备降落，错误代码：', e)
    dj.land()  # 降落


dj.close()  # 释放飞机资源