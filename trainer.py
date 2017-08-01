#!/usr/bin/python

import cv2, os, sys
import numpy as np
from PIL import Image

classifier_file = 'haarcascade_frontalface_default.xml'

images_ext = ['.png', '.jpg', '.jpeg']

def getFacesForTrainer(imagesPath, detector):
    faces = []
    identifiers = []

    images = [os.path.join(imagesPath, img) for img in os.listdir(imagesPath) if os.path.splitext(img)[1] in images_ext]

    count=1
    for image in images:
        pilImage = Image.open(image).convert('L')
        imageNp = np.array(pilImage, 'uint8')

        _faces = detector.detectMultiScale(imageNp)

        for (x, y, w, h) in _faces:
            faces.append(imageNp[y: y+h, x: x+w])
            identifiers.append(count)
            count=count+1

    return faces, identifiers



def setTrainer(imagesPath, outputDirectory, fileName='trainer'):
    detector, recognizer = getDectectorAndReconigzer()

    outputFile = os.path.join(outputDirectory, '{0}.yml'.format(fileName))

    faces, identifiers = getFacesForTrainer(imagesPath, detector)
    recognizer.train(faces, np.array(identifiers))
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
