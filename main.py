import numpy as np
import cv2 as cv
import argparse
path = "testy/test2.mp4"

cap = cv.VideoCapture(path)
scale = 30

while cap.isOpened():
    ret, frame = cap.read()
    select = cv.imwrite("select.jpg", frame)
    break

cap.release()
img = cv.imread('select.jpg')
r = cv.selectROI("Select an area then press enter", img)
print(r)
if cv.waitKey == ord('\n'):
    cv.destroyAllWindows


cap = cv.VideoCapture(path)



while cap.isOpened():
    ret, frame = cap.read()
    height = int(frame.shape[0]*scale/100)
    width = int(frame.shape[1]*scale/100)
    dim = (height, width)

    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break


    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    cv.imshow('frame', gray)
    if cv.waitKey(1) == ord('q'):
        break


cv.destroyAllWindows()
cap.release()
