1) connect the pi to computer using micro usb cable and plug the adapter on the bracket
2)open remote desktop connection(takes 1-2 min) and type user id ="pi" password ="raspberry"
3) to add your face for recognition open facedetect.py type in your name and face the camera it will take 30 images of you and show the image on the screen
4) after adding your face run facetrain.py for training the recognizer
5) now run face recognizer.py (need 1-2 min) and face the camera, the screen will show you how much you match with your image and the solenoid lock will open




#NOTE 

haarcascade_frontalface_default.xml is precreated file for object detection by Rainer Lienhart.
To learn more about haarcascade:-https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html