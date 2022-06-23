from yolov5_new.detect import DetectApi

model = DetectApi(weights=['.\\yolov5_new\\weights\\best.pt'], nosave=False)
## 目标检测
img_path = '.\\img\\20.jpg'  #待检测图片
res = model.detect(source=img_path)
print(res)  #显示检测结果