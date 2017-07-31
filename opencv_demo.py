#!/usr/bin/pythoon

# OpenCV 2 - Face Live Detection Demo

import numpy as np
import cv2

detector = cv2.CascadeClassifier('haarscade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

while(True):
   ret, img = cap.read()
   gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
   faces = detector.detectMultiScale(gray, 1.3, 5)

   for (x,y,z,h) in faces:
       cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0, 0), 2)

   cv2.imshow('OpenCV Face Live Detection Demo', img)
   
   if cv2.waitKey(1) & 0xFF == ord('q'):
       break

cap.release()
cv2.destroyAllWindows()
