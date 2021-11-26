import cv2                                      #Importing 
from picamera.array import PiRGBArray           #diffrent   
from picamera import PiCamera                   #libraries
import numpy as np                              #from
import pickle                                   #opencv
import RPi.GPIO as GPIO                         #and
from time import sleep                          #raspberry pi
relay_pin = [26] #variable for gpio pin 26
GPIO.setmode(GPIO.BCM)  #for representing the gpio pins number as bcm
GPIO.setup(relay_pin, GPIO.OUT) #setting pin 26 as op
GPIO.output(relay_pin, 1) #setting gpio pin high
with open('labels', 'rb') as f: #loading pickle file containing dictonary ie dict
    dict = pickle.load(f)
    f.close()
camera = PiCamera() #variabe set for picamera
camera.resolution = (640, 480) #frame resolution
camera.framerate = 30 # framerate
rawCapture = PiRGBArray(camera, size=(640, 480)) #frame size
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml") #defining classifier file
recognizer = cv2.face.LBPHFaceRecognizer_create() #face recog file incl in openCV
recognizer.read("trainer.yml") #read trained image file
font = cv2.FONT_HERSHEY_SIMPLEX 
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True): #converting image to grayscale for recognization
    frame = frame.array
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor = 1.5, minNeighbors = 5)
    for (x, y, w, h) in faces: #matching confidance of person from saved trained image
        roiGray = gray[y:y+h, x:x+w]
        id_, conf = recognizer.predict(roiGray)
        for name, value in dict.items():
            if value == id_:
                print(name)
        if conf <= 70: #condition for confidance <= 70 wil open the lock 
            GPIO.output(relay_pin,0) #giving gpio 26 pin to gnd to open lock
            print("opening lock")
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2) #creating box on face
            cv2.putText(frame, name + str(conf), (x, y), font, 2, (0, 0 ,255), 2,cv2.LINE_AA) # displays name of person on top of box
        else: #if face doese't match do nothing
            GPIO.output(relay_pin, 1)
    cv2.imshow('frame', frame) # shows frame ie stream from camera
    key = cv2.waitKey(1) #for cont live feed from camera
    rawCapture.truncate(0)
    if key == 27: #press ESC to get out of loop
        print("cleanup") 
        GPIO.cleanup() #to clean the signal on all gpio pins
        break
cv2.destroyAllWindows() # stop all the processing of CV2
