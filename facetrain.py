import os                    #Importing
import numpy as np           #library
from PIL import Image        #from
import cv2                   #cv2
import pickle                #and pi
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml") #to read classifier file in same dir
recognizer = cv2.face.LBPHFaceRecognizer_create() #local binary patterns histograms face recognizer included in opencv package
baseDir = os.path.dirname(os.path.abspath(__file__)) #path of current working dir to move images
imageDir = os.path.join(baseDir, "images")
currentId = 1
labelIds = {}
yLabels = []
xTrain = []
for root, dirs, files in os.walk(imageDir): #searching img dir and convert images to NumPy array
    print(root, dirs, files)
    for file in files: 
        print(file)
        if file.endswith("png") or file.endswith("jpg"): #checking for jpg and png file in saved img dir
            path = os.path.join(root, file)
            label = os.path.basename(root)
            print(label)
            if not label in labelIds:
                labelIds[label] = currentId
                print(labelIds)
                currentId += 1
            id_ = labelIds[label]
            pilImage = Image.open(path).convert("L")
            imageArray = np.array(pilImage, "uint8")
            faces = faceCascade.detectMultiScale(imageArray, scaleFactor=1.1, minNeighbors=5)
            for (x, y, w, h) in faces: #checking dir for image to see if we have right image to train 
                roi = imageArray[y:y+h, x:x+w] # face detection
                xTrain.append(roi)
                yLabels.append(id_)
with open("labels", "wb") as f:
    pickle.dump(labelIds, f)
    f.close()
recognizer.train(xTrain, np.array(yLabels)) #train data and save file as trainer.yml which will be used to read the traied data for particular image in face recog.
recognizer.save("trainer.yml")
print(labelIds)
