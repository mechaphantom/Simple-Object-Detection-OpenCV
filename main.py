import cv2
from tracker import *

cap = cv2.VideoCapture("road.mp4")

#Tracker Object
tracker = EuclideanDistTracker()
#This object tracking algorithm is called centroid tracking as it relies on the Euclidean distance between existing object centroids
#and new object centroids between subsequent frames in a video.

#Object Detector
#Gaussian mixture-based background/foreground segmentation algorithm.

object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

#history: Returns the number of last frames that affect the background model.
#varThreshold: Returns the variance threshold for the pixel-model match.
#The main threshold on the squared Mahalanobis distance to decide if the sample is well described by the background model or not.

while True:
    ret, frame = cap.read()
    height, width, _ = frame.shape
    
    #Region Of Interest (We identify a specific area we want to work in.)
    #(Not necessary. You can work with the whole video if you wish.)
    roi = frame[320:720, 0:1280]
    
    #Object Detection
    mask = object_detector.apply(roi)
    
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    #THRESH_BINARY: The function transforms a grayscale image to a binary image.   
    #dst(x,y)={ maxval if src(x,y)>thresh
    #        ={ 0      if otherwise
    
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #Contours are defined as the line joining all the points along the boundary of an image that are having the same intensity.
    #RETR_TREE: It retrieves all the contours and creates a full family hierarchy list.
    #CHAIN_APPROX_SIMPLE: returns only the endpoints that are necessary for drawing the contour line. 
    detections = []
    
    for i in contours:
        area = cv2.contourArea(i)
        if area > 75: #We control whether the size of the white masked area is greater than 75.
            x, y, w, h = cv2.boundingRect(i)
            detections.append([x, y, w, h])
            #cv2.rectangle(roi, (x,y), (x+w, y+h), (0,255,0), 3)

    # 2. Object Tracking
    boxes_ids = tracker.update(detections)
    for box_id in boxes_ids:
        x, y, w, h, id = box_id
        cv2.putText(roi, str(id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)
    
    cv2.imshow("ROI", roi)
    cv2.imshow("Video", frame)
    cv2.imshow("Mask", mask)
    
    key = cv2.waitKey(30)
    if key == 27: # 's'
        break
        
cap.release()
cv2.destroyAllWindows()
