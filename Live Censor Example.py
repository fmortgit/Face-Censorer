import numpy as np
import cv2

from FaceCensorer import FaceCensorer

cap = cv2.VideoCapture(0)
censor = FaceCensorer(10)

while True:
    ret, frame = cap.read()

    inputImg = cv2.flip(frame, 1)

    outputImg = censor.censorImage(inputImg)

    cv2.imshow("", outputImg)

    if cv2.waitKey(1) == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()