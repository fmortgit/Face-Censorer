import numpy as np
import cv2



class FaceCensorer():
    def __init__(self, squareSize):
        self.squareSize = squareSize
        self.faceClassifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def censorImage(self, image):
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faceList = self.faceClassifier.detectMultiScale(grayImage)

        if len(faceList) != 0:  # check if there is at least one face present

            for face in faceList:

                # face is an ndarray with [[xPos, yPos, width, height]]
                values = face

                x, y, width, height = values[0], values[1], values[2], values[3]


                for i in range(x, x+width, self.squareSize):
                    for j in range(y, y+height, self.squareSize):

                        roi = image[j:j+self.squareSize, i:i+self.squareSize]

                        r = np.mean(roi[:,:,0])
                        g = np.mean(roi[:,:,1])
                        b = np.mean(roi[:,:,2])
                        avgColour = (r,g,b)

                        cv2.rectangle(image, (i,j), (i+self.squareSize, j+self.squareSize), avgColour, -1)

        return image

    @property
    def squareSize(self):
        return self._squareSize

    @squareSize.setter
    def squareSize(self, value):
        self._squareSize = value

    @property
    def faceClassifier(self):
        return self._faceClassifier

    @faceClassifier.setter
    def faceClassifier(self, value):
        self._faceClassifier = value
