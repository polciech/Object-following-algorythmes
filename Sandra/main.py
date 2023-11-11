import cv2 as cv
import numpy as np
import tkinter as tk
from tkinter import messagebox

class ObjectTracker:
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv.VideoCapture(self.video_path)
        self.root = tk.Tk()
        self.root.title("Object Tracking GUI")

        self.roi_hist = None
        self.r = None

        self.term_criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)

        self.create_gui()

    def create_gui(self):
        select_button = tk.Button(self.root, text="Select Object", command=self.select_object)
        select_button.pack(pady=10)

        track_button = tk.Button(self.root, text="Track Object", command=self.track_object)
        track_button.pack(pady=10)

        exit_button = tk.Button(self.root, text="Exit", command=self.exit_program)
        exit_button.pack(pady=10)

    def select_object(self):
        ret, frame = self.cap.read()
        cv.imwrite("select.jpg", frame)
        img = cv.imread('select.jpg')
        roi = cv.selectROI("Select an area then press enter", img)
        cv.destroyAllWindows()

        # Set up the initial tracking window
        x, y, w, h = int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])
        roi = frame[y:y+h, x:x+w]
        roi_hsv = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
        mask = cv.inRange(roi_hsv, np.array((0., 60., 32.)), np.array((180., 255., 255.)))

        self.roi_hist = cv.calcHist([roi_hsv], [0], mask, [180], [0, 180])
        cv.normalize(self.roi_hist, self.roi_hist, 0, 255, cv.NORM_MINMAX)
        self.r = (x, y, w, h)

        messagebox.showinfo("Info", "Object selected successfully.")

    def track_object(self):
        if self.roi_hist is None:
            messagebox.showinfo("Info", "Select an object first.")
            return

        while self.cap.isOpened():
            ret, frame = self.cap.read()

            if not ret:
                messagebox.showinfo("Info", "Can't receive frame (stream end?). Exiting ...")
                self.exit_program()
                return

            hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
            dst = cv.calcBackProject([hsv], [0], self.roi_hist, [0, 180], 1)

            # Convert dst to 8-bit unsigned integer
            dst = np.uint8(dst)

            # Apply meanshift to get the new location
            ret, self.r = cv.meanShift(dst, self.r, self.term_criteria)

            # Draw the rectangle on the image
            x, y, w, h = self.r
            frame = cv.rectangle(frame, (x, y), (x+w, y+h), 255, 2)

            cv.imshow('frame', frame)

            if cv.waitKey(30) == ord('q'):
                self.exit_program()
                return

    def exit_program(self):
        self.cap.release()
        cv.destroyAllWindows()
        self.root.destroy()

if __name__ == "__main__":
    video_path = "Ball2.mp4"
    tracker = ObjectTracker(video_path)
    tk.mainloop()
