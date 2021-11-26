import cv2 #calling cv2 library for image processing
from picamera.array import PiRGBArray #pi camera rgb and gray scale
from picamera import PiCamera #pi camera library
import numpy as np #calling cv2 library to converting images to numerical array
import os #dir Handeling
import sys
camera = PiCamera() #variable
camera.resolution = (640, 480) #resolution of camera
camera.framerate = 30 #framerate
rawCapture = PiRGBArray(camera, size=(640, 480)) #frame size
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml") #cv2 harrcascade face detection lib
name = input("What's his/her Name? ") #variable for name
dirName = "./images/" + name # directory to save img
print(dirName) #to show dir name in shell
if not os.path.exists(dirName): #logic for directory exists or not
    os.makedirs(dirName)
    print("Directory Created")
else:
    print("Name already exists")
    sys.exit()
count = 1
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True): #no of images to capture
    if count > 30:
        break
    frame = frame.array
    gray =cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor = 1.5, minNeighbors = 5) #recognize faces
    for (x,y,w,h) in faces: #converting img to greyscale and save name of image as given by user 
        roiGray = gray[y:y+h, x:x+w]
        fileName = dirName + "/" +name +str(count) + ".jpg"
        cv2.imwrite(fileName, roiGray)
        cv2.imshow("face",roiGray)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        count+=1
    cv2.imshow('frame', frame) #display the frame
    key = cv2.waitKey(1)
    rawCapture.truncate(0)
    if key == 27: #ESC to exit from code
        break
cv2.destroyAllWindows()  #To clear the image Frame window
    
