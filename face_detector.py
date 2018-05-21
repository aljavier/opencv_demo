#!/usr/bin/python

import cv2
import sys

if len(sys.argv) <= 1:
    print("Please provide the image path and name of the subject to look for.")
    sys.exit(-1)

trainer_file = 'trainer.yml'
classifier_file = 'haarcascade_frontalface_default.xml'
max_confidence = 55 # This value or less

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(trainer_file)
classifier = cv2.CascadeClassifier(classifier_file)

font = cv2.FONT_HERSHEY_SIMPLEX
fontscale = 1
fontcolor = (255,255,255)

while(True):
    img = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)

    faces = classifier.detectMultiScale(img, 1.3, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0, 0), 2)
        identifier, confidence = recognizer.predict(img[y: y+h, x: x+w])
        
        text = "{0:.2f} ".format(confidence)
        if confidence <= max_confidence:
           text += "Found!"
        else:
           text += "Not sure."

        cv2.putText(img,text,(x+2,y+h-7),font,fontscale,fontcolor)

    cv2.imshow("Search results", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
