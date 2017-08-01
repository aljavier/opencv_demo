#!/usr/bin/python

import cv2, os, sys
import numpy as np
from PIL import Image

classifier_file = 'haarcascade_frontalface_default.xml'

MAX_FACES = 1

def getFacesForTrainer(imagePath, detector):
    faces = []

    pilImage = Image.open(imagePath).convert('L')
    imageNp = np.array(pilImage, 'uint8')

    _faces = detector.detectMultiScale(imageNp)

    for (x, y, w, h) in _faces[:MAX_FACES]:
        faces.append(imageNp[y: y+h, x: x+w])

    return faces



def setTrainer(imagePath, outputDirectory, fileName='trainer'):
    detector, recognizer = getDectectorAndReconigzer()

    outputFile = os.path.join(outputDirectory, '{0}.yml'.format(fileName))
    
    faces = getFacesForTrainer(imagePath, detector)
    recognizer.train(faces, np.array(7))
    recognizer.save(outputFile)

    print('{0} faces detected! Info stored in {1}'.format(len(faces), outputFile))



def getDectectorAndReconigzer():
    if not os.path.isfile(classifier_file):
        print("%s does not exits in the current directory!" % classifier_file)
        return
   
    recognizer = cv2.createLBPHFaceRecognizer()
    detector = cv2.CascadeClassifier(classifier_file)

    return  detector, recognizer


if __name__ == '__main__':
    if len(sys.argv) <= 1:
        print('You must provide the full path of the image!')
        sys.exit(-1)

    setTrainer(sys.argv[1], '.')





