from detect import DetectApi
import time , operator

# model = DetectApi(weights=['weights\\best.pt'], nosave=False)



def get_qi_info(model,img_path):
    result = dict()
    res1 = model.detect(source=img_path,conf_thres=0.3)
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
    print(res)
    # print(res)
    qi = {}
    for r in res:
        if r['label'] == 'qi':
            qi = r
            break
    try:
        if qi:
            """w=30"""
            # f = (p*d)/w #f:焦距，p实物在图片中的像素，d:摄像头距离实物的距离，w:物体实际宽度cm
            f = 1020.47
            # dis = (w*f)/p1 # w:物体实际宽度cm ，f：焦距, p1物体在图片中的像素宽度
            dis = (30*f)/(qi['x2y2'][0]-qi['x1y1'][0])
            result['dis_forward'] = dis # 距离旗子的距离
            dis_cm = 30/(qi['x2y2'][0]-qi['x1y1'][0]) # 每像素多少厘米，需要用相似三角形计算
            if int(qi['x1y1'][0]) < 10:
                result['code'] = 'action'
                result['direction'] = 'left'
                result['distance'] = 30
                result['finish'] = False
                result['msg'] ='旗子在左边界，测量可能不准确，需向 左  飞行后重新拍照计算'
                return result
            elif int(qi['src_img_size'][1]) == (qi['x2y2'][0]):
                result['code'] = 'action'
                result['direction'] = 'right'
                result['distance'] = 30
                result['finish'] = False
                result['msg'] ='旗子在右边界，测量可能不准确，需向  右  飞行重新拍照计算'
                return result
            diff = (qi['src_img_size'][1]-qi['x2y2'][0]-qi['x1y1'][0])/2
            diff = int(diff*dis_cm)
            print('diff:: ',diff)
            if abs(diff)<20:
                result['finish'] = True
                result['code'] = 'no action'
                result['msg'] = '偏移居里路小于20，无需调整位置'
            else:
                if diff>0:
                    result['direction'] = 'left'
                    result['msg'] = f'需向左飞{diff}'
                    # print('向左飞')
                else:
                    result['direction'] = 'right'
                    result['msg'] = f'需向右飞{diff}'
                result['code'] = 'action'
                result['distance'] = abs(diff)
                result['finish'] = True
                    # print('向右飞')
        else:
            print('未识别到旗')
            result['code'] = 'err'
            result['finish'] = False
            result['msg'] = '未识别到旗'
    except Exception as e:
        print(e)
        result['code'] = 'err'
        result['msg'] = str(e)
        result['finish'] = True

    return result



def get_quan_info(model,img_path):
    result = dict()
    res1 = model.detect(source=img_path,conf_thres=0.3)
    # res1 = model.detect(source='..\\tello_tt_yolov5\\img\\1.2.jpg',conf_thres=0.4)
    # res2 = model.detect(source='..\\tello_tt_yolov5\\img\\1.2-2.jpg',conf_thres=0.4)
    # res3 = model.detect(source='..\\tello_tt_yolov5\\img\\1.3-1.jpg',conf_thres=0.4)
    # res4 = model.detect(source='..\\tello_tt_yolov5\\img\\1.3-2.jpg',conf_thres=0.4)
    # res5 = model.detect(source='..\\tello_tt_yolov5\\img\\1.3-3.jpg',conf_thres=0.4)
    # print(res1)
    # print(res2)
    # print(res3)
    # print(res4)
    # print(res5)

    t3 = time.time()
    res = sorted(res1, key=operator.itemgetter('conf'), reverse=True)
    print(res)
    # print(res)
    quan = {}
    for r in res:
        if r['label'] == 'quan':
            quan = r
            break
    try:
        if quan:
            """w=48"""
            # f = (p*d)/w #f:焦距，p实物在图片中的像素，d:摄像头距离实物的距离，w:物体实际宽度cm
            f = 391.67
            # dis = (w*f)/p1 # w:物体实际宽度cm ，f：焦距, p1物体在图片中的像素宽度
            dis = (48*f)/(quan['x2y2'][0]-quan['x1y1'][0])
            result['dis_forward'] = dis # 距离圈的距离
            dis_cm = 48/(quan['x2y2'][0]-quan['x1y1'][0]) # 每像素多少厘米，需要用相似三角形计算
            if int(quan['x1y1'][0]) < 10:
                result['code'] = 'action'
                result['direction'] = 'left'
                result['distance'] = 30
                result['finish'] = False
                result['msg'] ='圈在左边界，测量可能不准确，需向 左  飞行后重新拍照计算'
                return result
            elif int(quan['src_img_size'][1]) == (quan['x2y2'][0]):
                result['code'] = 'action'
                result['direction'] = 'right'
                result['distance'] = 30
                result['finish'] = False
                result['msg'] ='圈在右边界，测量可能不准确，需向  右  飞行重新拍照计算'
                return result
            diff = (quan['src_img_size'][1]-quan['x2y2'][0]-quan['x1y1'][0])/2
            diff = int(diff*dis_cm)
            print('diff:: ',diff)
            if abs(diff)<20:
                result['finish'] = True
                result['code'] = 'no action'
                result['msg'] = '偏移居里路小于20，无需调整位置'
            else:
                if diff>0:
                    result['direction'] = 'left'
                    result['msg'] = f'需向左飞{diff}'
                    # print('向左飞')
                else:
                    result['direction'] = 'right'
                    result['msg'] = f'需向右飞{diff}'
                result['code'] = 'action'
                result['distance'] = abs(diff)
                result['finish'] = True
                    # print('向右飞')
        else:
            print('未识别到圈')
            result['code'] = 'err'
            result['finish'] = False
            result['msg'] = '未识别到圈'
    except Exception as e:
        print(e)
        result['code'] = 'err'
        result['msg'] = str(e)
        result['finish'] = True

    return result
