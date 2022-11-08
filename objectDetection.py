import cv2 as cv
import numpy as np
from tracker import *

# Create Tracker object

tracker= EuclideanDistTracker()

capture=cv.VideoCapture('../resources/videos/1.avi')

def rescaleFrame(frame, scale=0.75):
    width=int(frame.shape[1]*scale)
    height=int(frame.shape[0]*scale)
    dimensions=(width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)


# Object Detection using camera
object_detector=cv.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

while True: 
    isTrue, frame= capture.read()

    frame_resized=rescaleFrame(frame, 0.4)
    
    #region of interest
    roi=frame_resized[40:420, :730]

    # 1. Object Detection 
    mask=object_detector.apply(roi)
    # this line is to remove shadow, but this will then not 
    # cover players with black jersey since threshold of contour is very high!!
    # _, mask=cv.threshold(mask, 254, 255, cv.THRESH_BINARY) 

    contours, _= cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    detections=[]
    for cnt in contours:
        # Calculate area and remove all small elements
        area=cv.contourArea(cnt)
        if area > 100:
            # cv.drawContours(roi, [cnt], -1, (0, 255, 0), 2)
            x, y, w, h = cv.boundingRect(cnt)
            
            # print(x, y, w, h)
            detections.append([x,y,w,h])


    # 2. Object Tracking
    boxes_ids=tracker.update(detections)
    for box_id in boxes_ids:
        x, y, w, h, id = box_id
        cv.putText(roi, str(id), (x, y-15), cv.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
        cv.rectangle(roi,(x,y), (x+w, y+h), (0, 255, 0), 3) 



    
    cv.imshow('Video', frame_resized)
    cv.imshow('Mask', mask)
    cv.imshow('ROI', roi)

    if cv.waitKey(20) & 0xFF==ord('d'):
        break


capture.release()
cv.destroyAllWindows()
cv.waitKey(0)