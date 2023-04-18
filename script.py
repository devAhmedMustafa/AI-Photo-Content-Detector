import cv2 as cv
img = cv.imread('images/employee.png')
classnames = []
classfile = 'files/thing.names'

with open(classfile, 'rt') as f:
    classnames = f.read().rstrip('\n').split('\n')
#print(classnames)

p = 'files/frozen_inference_graph.pb'
v = 'files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'

net = cv.dnn_DetectionModel(p,v)
net.setInputSize(320,230)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5,127.5,127.5))
net.setInputSwapRB(True)

classIds, confs, bbox = net.detect(img, confThreshold=0.5)

#print(classIds, bbox)
for classId, confidence, box in zip(classIds.flatten(), confs.flatten(),bbox):
    cv.rectangle(img,box,color=(0,255,0),thickness=1)
    cv.putText(
        img, classnames[classId-1],
        (box[0]+ 10, box[1]+ 20),
        cv.FONT_HERSHEY_COMPLEX,1,(0,0,255), thickness=1
    )

cv.imshow('Program', img)
cv.waitKey(0)