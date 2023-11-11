import cv2 as cv
import numpy as np

path = "Ball2.mp4"

cap = cv.VideoCapture(path)

# Read the first frame and select the object to track
ret, frame = cap.read()
select = cv.imwrite("select.jpg", frame)
img = cv.imread('select.jpg')
r = cv.selectROI("Select an area then press enter", img)
cv.destroyAllWindows()

# Set up the initial tracking window
roi = frame[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
roi_hsv = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
mask = cv.inRange(roi_hsv, np.array((0., 60., 32.)), np.array((180., 255., 255.)))

roi_hist = cv.calcHist([roi_hsv], [0], mask, [180], [0, 180])
cv.normalize(roi_hist, roi_hist, 0, 255, cv.NORM_MINMAX)

term_criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    dst = cv.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

    # Apply meanshift to get the new location
    ret, r = cv.meanShift(dst, r, term_criteria)

    # Draw the rectangle on the image
    x, y, w, h = r
    frame = cv.rectangle(frame, (x, y), (x+w, y+h), 255, 2)

    cv.imshow('frame', frame)

    if cv.waitKey(30) == ord('q'):
        break

cv.destroyAllWindows()
cap.release()
