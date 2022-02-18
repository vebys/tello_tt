from detect import DetectApi
import time , operator
t1 = time.time()
model = DetectApi(weights=['runs\\train\\exp12\\weights\\best.pt'], nosave=False)
t2 = time.time()
res1 = model.detect(source='..\\tt_data\\test\\img\\1.45-2.jpg',conf_thres=0.4)
# res2 = model.detect(source='..\\tt_data\\test\\img\\1.2-2.jpg',conf_thres=0.4)
# res3 = model.detect(source='..\\tt_data\\test\\img\\1.3.jpg',conf_thres=0.4)
# res4 = model.detect(source='..\\tt_data\\test\\img\\1.45-1.jpg',conf_thres=0.4)
# res5 = model.detect(source='..\\tt_data\\test\\img\\1.45-2.jpg',conf_thres=0.4)
# print(res1)
# print(res2)
# print(res3)
# print(res4)
# print(res5)
t3 = time.time()
res = sorted(res1, key=operator.itemgetter('conf'), reverse=True)
# print(res)
# print(res)
qi = {}
for r in res:
    if r['label'] == 'qi':
        qi = r
        break
try:
    if qi:
        # f = (p*d)/w #f:焦距，p实物在图片中的像素，d:摄像头距离实物的距离，w:物体实际宽度cm
        f = 1020.47
        # dis = (w*f)/p1 # w:物体实际宽度cm ，f：焦距, p1物体在图片中的像素宽度
        dis = (30*f)/(qi['x2y2'][0]-qi['x1y1'][0])
        dis_cm = 30/(qi['x2y2'][0]-qi['x1y1'][0]) # 每像素多少厘米，需要用相似三角形计算
        if qi['x1y1'][0] == 0:
            raise NameError('旗子在左边界，测量可能不准确，需向 左  飞行后重新拍照计算')
        elif qi['src_img_size'][1] == qi['x2y2'][0]:
            raise NameError('旗子在右边界，测量可能不准确，需向  右  飞行重新拍照计算')
        diff = (qi['src_img_size'][1]-qi['x2y2'][0]-qi['x1y1'][0])/2
        diff = int(diff*dis_cm)
        print('diff:: ',diff)
        if abs(diff)<20:
            raise NameError('偏移居里路小于20，无需调整位置')
        else:
            if diff>0:
                print('向左飞')
            else:
                print('向右飞')
    else:
        print('未识别到旗')
        raise NameError('未识别到旗')
except Exception as e:
    print(e)
print('t2-t1', t2-t1, '  t3-t2', t3-t2)