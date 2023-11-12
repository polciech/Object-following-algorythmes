import cv2 as cv
import numpy as np
import tkinter as tk
from tkinter import messagebox, ttk, filedialog
import time


class ObjectTracker:
    def __init__(self):
        self.video_path = None
        self.cap = None
        self.root = tk.Tk()
        self.root.title("Object Tracking GUI")

        self.roi_hist = None
        self.r = None
        self.path = []

        self.term_criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)
        self.tracking_algorithm = tk.StringVar(value="MeanShift")

        self.create_gui()

    def create_gui(self):
        select_button = tk.Button(self.root, text="Select Object", command=self.select_object)
        select_button.pack(pady=10)

        tracking_label = tk.Label(self.root, text="Select Tracking Algorithm:")
        tracking_label.pack()

        tracking_combobox = ttk.Combobox(self.root, textvariable=self.tracking_algorithm,
                                         values=["MeanShift", "CamShift"])
        tracking_combobox.pack()

        select_video_button = tk.Button(self.root, text="Select Video", command=self.select_video)
        select_video_button.pack(pady=10)

        track_button = tk.Button(self.root, text="Track Object", command=self.track_object)
        track_button.pack(pady=10)

        exit_button = tk.Button(self.root, text="Exit", command=self.exit_program)
        exit_button.pack(pady=10)

    def select_object(self):
        if self.video_path is None:
            messagebox.showinfo("Info", "Select a video first.")
            return

        ret, frame = self.cap.read()
        cv.imwrite("select.jpg", frame)
        img = cv.imread('select.jpg')
        roi = cv.selectROI("Select an area then press enter", img)
        cv.destroyAllWindows()

        x, y, w, h = int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])
        roi = frame[y:y + h, x:x + w]
        roi_hsv = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
        mask = cv.inRange(roi_hsv, np.array((0., 60., 32.)), np.array((180., 255., 255.)))

        self.roi_hist = cv.calcHist([roi_hsv], [0], mask, [180], [0, 180])
        cv.normalize(self.roi_hist, self.roi_hist, 0, 255, cv.NORM_MINMAX)
        self.r = (x, y, w, h)
        self.path = [(x + w // 2, y + h // 2)]

        messagebox.showinfo("Info", "Object selected successfully.")

    def select_video(self):
        file_path = filedialog.askopenfilename(title="Select Video File", filetypes=[("Video Files", "*.mp4;*.avi")])
        if file_path:
            self.video_path = file_path
            self.cap = cv.VideoCapture(self.video_path)
            messagebox.showinfo("Info", f"Selected video: {self.video_path}")

    def track_object(self):
        if self.roi_hist is None:
            messagebox.showinfo("Info", "Select an object first.")
            return

        if self.video_path is None:
            messagebox.showinfo("Info", "Select a video first.")
            return

        while self.cap.isOpened():
            ret, frame = self.cap.read()

            if not ret:
                self.exit_program(frame)
                break

            hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
            dst = cv.calcBackProject([hsv], [0], self.roi_hist, [0, 180], 1)
            dst = np.uint8(dst)

            if self.tracking_algorithm.get() == "MeanShift":
                ret, self.r = cv.meanShift(dst, self.r, self.term_criteria)
            elif self.tracking_algorithm.get() == "CamShift":
                ret, self.r = cv.CamShift(dst, self.r, self.term_criteria)

            x, y, w, h = self.r
            self.r = (x, y, w, h)

            if self.is_out_of_bounds(x, y, w, h, frame.shape[1], frame.shape[0]):
                self.exit_program(frame)
                break

            if self.tracking_algorithm.get() == "CamShift":
                pts = cv.boxPoints(ret)
                pts = np.int0(pts)
                frame = cv.polylines(frame, [pts], True, 255, 2)
            else:
                frame = cv.rectangle(frame, (x, y), (x + w, y + h), 255, 2)

            self.path.append((x + w // 2, y + h // 2))
            self.draw_path(frame)

            cv.imshow('frame', frame)

            if cv.waitKey(30) == ord('q'):
                self.exit_program(frame)
                break

    def is_out_of_bounds(self, x, y, w, h, max_width, max_height):
        return x < 0 or y < 0 or x + w > max_width or y + h > max_height

    def draw_path(self, frame):
        if len(self.path) > 1:
            for i in range(1, len(self.path)):
                cv.line(frame, self.path[i - 1], self.path[i], (0, 255, 0), 2)

    def exit_program(self, frame):
        if self.cap.isOpened():
            total_frames = int(self.cap.get(cv.CAP_PROP_FRAME_COUNT))
            before_last_frame_timestamp = int(total_frames - 2)
            self.cap.set(cv.CAP_PROP_POS_FRAMES, before_last_frame_timestamp)

            ret, frame = self.cap.read()
            if ret:
                self.draw_path(frame)
                cv.imwrite("path_with_tracking.jpg", frame)
            else:
                messagebox.showinfo("Info", "Object went out of bounds. Exiting ...")

            if self.cap.isOpened():
                total_frames = int(self.cap.get(cv.CAP_PROP_FRAME_COUNT))
                before_last_frame_timestamp = int(total_frames - 2)
                self.cap.set(cv.CAP_PROP_POS_FRAMES, before_last_frame_timestamp)

                ret, frame = self.cap.read()
                if ret:
                    self.draw_path(frame)
                    cv.imwrite("path_with_tracking.jpg", frame)
                else:
                    messagebox.showinfo("Info", "Error capturing the frame")

                self.cap.release()
                cv.destroyAllWindows()
                self.root.destroy()


if __name__ == "__main__":
    tracker = ObjectTracker()
    tk.mainloop()
