from ultralytics import YOLO
import cv2
import math
from sort import *
import numpy as np

def start():
    model = YOLO("../YOLO_weights/yolov8l.pt")
    names = model.model.names
    limits = [50, 420, 1250, 420]
    tracker = Sort(max_age=10, min_hits=3, iou_threshold=0.3)
    totalCount = []

    cap = cv2.VideoCapture("second/traffic_video.mp4")
    mask = cv2.imread("second/mask_1.png")
    mask = cv2.resize(mask, (1280, 720))
    while True:
        success, img = cap.read()
        img_with_mask = cv2.bitwise_and(img, mask)
        results = model(img_with_mask, stream=True)

        detections = np.empty((0,5))

        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1,y1,x2,y2 = box.xyxy[0]
                #x1,y1,w,h = box.xywn[0]
                bbox = int(x1), int(y1), int(x2), int(y2) 
                conf = math.ceil((box.conf[0]*100)) / 100
                cls = int(box.cls[0])
                class_name = names[cls]
                if (class_name == "car" or class_name == "motorbike" or class_name == "bus" or class_name == "truck") and conf >= 0.6:
                    currentArr = np.array([bbox[0],bbox[1],bbox[2],bbox[3],conf])
                    detections = np.vstack((detections, currentArr))
        
        resultTracker = tracker.update(detections) 
        cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0,255,255), 4)
        for res in resultTracker:
            x1,x2,y1,y2,Id = res
            x1,x2,y1,y2 = int(x1), int(y1), int(x2), int(y2) 
            cv2.rectangle(img, pt1=(x1,y1), pt2=(x2,y2), color=(0,0,255), thickness=3)      
            cv2.putText(img, f"{int(Id)} {conf}",(max(0,x1)+10,max(0,y1)+10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            print(res)

            cx, cy = x2-((x2-x1)//2), y2-((y2-y1)//2)
            cv2.circle(img, (cx,cy), 5, (255,0,255),cv2.FILLED)

            if limits[0]<cx<limits[2] and limits[1]-20<cy<limits[3]+20:
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (9, 230, 105), 4)
                if totalCount.count(Id) == 0:
                    totalCount.append(Id)
        
        cv2.putText(img, f"Count: {len(totalCount)}", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 4, cv2.FILLED)


        cv2.imshow("Image", img)
        cv2.waitKey(1)

start()