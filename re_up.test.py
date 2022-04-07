from img.tello_sdk_stand import *
dj = Start()


try:
    """起飞"""
    # dj.led_obj.set_mled_sc()  # 关闭LED灯，节约用电
    dj.take_off()  # 起飞
    time.sleep(1)

    dj.curve(x1=100, y1=100, z1=20, x2=200, y2=0, z2=40, speed=30)

    dj.curve(x1=-100, y1=-100, z1=20, x2=-180, y2=0, z2=40, speed=30)



    dj.forward(220)
    dj.down(90)
    dj.reverse(360)
    dj.up(90)
    dj.forward(230)
    dj.down(90)
    dj.back(100)
    dj.up(90)
    dj.forward(100)
    dj.right(140)
    dj.land()
except Exception as e:
    print('飞行异常准备降落，错误代码：', e)
    dj.land()  # 降落


dj.close()  # 释放飞机资源