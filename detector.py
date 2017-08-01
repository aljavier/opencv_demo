#!/usr/bin/python

import numpy as np
import cv2
import sys
import haslish

if len(sys.argv) <= 2:
    print("Debe proveer la imagen y nombre del sospechoso!")
    sys.exit(-1)

name = sys.argv[2]
print("Buscando a {0}...".format(name))

trainer_file = 'trainer.yml'
classifier_file = 'haarcascade_frontalface_default.xml'

recognizer = cv2.createLBPHFaceRecognizer()
recognizer.load(trainer_file)
classifier = cv2.CascadeClassifier(classifier_file)

font = cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_SIMPLEX, 1, 1, 0, 1, 1)

while(True):
    img = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
    #ret, im = img.read()

    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = classifier.detectMultiScale(img, 1.3, 5) # gray por img

    text = "Desconocido"

    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0, 0), 2)
        identier, confidence = recognizer.predict(img[y: y+h, x: x+w]) # gray por img

        if(confidence < 50):
            text = "{0}".format(confidence)

        cv2.cv.PutText(cv2.cv.fromarray(img),text,(x, y+h),font,255)

    cv2.imshow("{0} encontrado!".format(name), img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
