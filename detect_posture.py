import sys
from collections import UserList
import numpy as np
from matplotlib import pyplot as plt
import cv2
import tkinter as tk
from tkinter import messagebox


class LimitedList(UserList):
    """
    Custom list class for numbers

    The size of list can be limited.
    Store the maximum and minimum from the first.
    """
    def __init__(self, initlist=None):
        if initlist is not None and not all([type(e) is int or type(e) is float for e in initlist]):
            ValueError("invalid args for LimitedList.")
        super().__init__(initlist)
        self._maximum = max(self.data) if self.data else 0
        self._minimum = min(self.data) if self.data else 0
    
    def append(self, item):
        if type(item) is not int or type(item) is not float:
            ValueError(f"invalid argument for LimitedList: {item}")
        super().append(item)
    
    def insert(self, i, item):
        if type(item) is not int or type(item) is not float:
            ValueError(f"invalid argument for LimitedList: {item}")
        super().insert(i, item)
    
    def extend(self, iterable):
        if not all([type(e) is int or type(e) is float for e in iterable]):
            ValueError(f"invalid args for LimitedList.")
        super().extend(iterable)

    def max(self):
        return max((self._maximum, max(self.data)))
    
    def min(self):
        return min((self._minimum, min(self.data)))

    def limit(self, limit=10000):
        self._maximum = max(self.data) if self.data else 0
        self._minimum = min(self.data) if self.data else 0
        if len(self.data) > limit:
            self.data = self.data[-limit:]


WINDOW_NAME = "capture"     # Videcapute window name
CAP_FRAME_WIDTH = 640       # Videocapture width
CAP_FRAME_HEIGHT = 480      # Videocapture height
CAP_FRAME_FPS = 30          # Videocapture fps (depends on user camera)

DEVICE_ID = 1               # Web camera id (0 is maybe built-in camera)

EYE_HEIGHTS = LimitedList()         # History of eye height
SMA_EYE_HEIGHTS = LimitedList()     # History of the simple moving average (SMA) of eye hight

SMA_SEC = 60                        # SMA seconds
SMA_N = SMA_SEC * CAP_FRAME_FPS     # SMA n

PLOT_NUM = 20                   # Plot points number
PLOT_DELTA = 1/CAP_FRAME_FPS    # Step of X axis

Z = 45                  # (cm) Distance from PC to face 
D = 3                   # (cm) Limit of lowering eyes


def count_camera_connection(limit=10):
    """
    Count the connection between any cameras and the PC.
    """
    print("searching cameras...")
    valid_cameras = []

    for camera_number in range(limit):
        cap_cam = cv2.VideoCapture(camera_number, cv2.CAP_DSHOW)
        if cap_cam.isOpened():
            valid_cameras.append(camera_number)

    print(len(valid_cameras), "cameras available now.")
    return len(valid_cameras)


def moving_average(data, n):
    """
    Return simple moving average.
    """
    if len(data) < n:
        raise ValueError
    result = []
    for i in range(n-1, len(data)):
        total = 0
        for j in range(n):
            total += data[i-j]
        result.append(total/n)
    return result


def add_moving_average(smas, data, n):
    """
    Add the latest simple moving average.
    """
    if len(data) < n:
        raise ValueError
    total = 0
    for i in range(n):
        total += data[-1-i]
    smas.append(total/n)


if __name__ == '__main__':
    # Not show tkinter window
    root = tk.Tk()
    root.iconify()

    # Count the number of cameras.
    # If there are no cameras available, end program.
    if not count_camera_connection():
        sys.exit(1)

    # Set cascade
    cascade_file = "haarcascade_eye.xml"
    cascade = cv2.CascadeClassifier(cascade_file)

    # Capture setup
    cap = cv2.VideoCapture(DEVICE_ID, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAP_FRAME_FPS)

    # Prepare windows
    cv2.namedWindow(WINDOW_NAME)

    # Plot setup
    ax = plt.subplot()
    graph_x = np.arange(0, PLOT_NUM*PLOT_DELTA, PLOT_DELTA)
    eye_y = [0] * PLOT_NUM
    smas_eye_y = [0] * PLOT_NUM
    eye_lines, = ax.plot(graph_x, eye_y, label="realtime")
    smas_eye_lines, = ax.plot(graph_x, smas_eye_y, label="SMA")
    ax.legend()

    while cap.isOpened():
        ret, frame = cap.read()

        img = frame
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect human eyes
        eyes = cascade.detectMultiScale(img_gray, minSize=(30, 30))

        # Mark on the detected eyes
        for (x, y, w, h) in eyes:
            color = (255, 0, 0)
            line_width = 3
            cv2.rectangle(img_gray, (x, y), (x+w, y+h), color, thickness=line_width)
        
        # Store eye heights
        if len(eyes) > 0:
            eye_average_height = CAP_FRAME_HEIGHT - sum([y for _, y, _, _ in eyes]) / len(eyes)
            EYE_HEIGHTS.append(eye_average_height)

            if len(EYE_HEIGHTS) == SMA_N:
                SMA_EYE_HEIGHTS = LimitedList(moving_average(EYE_HEIGHTS, SMA_N))
            elif len(EYE_HEIGHTS) > SMA_N:
                add_moving_average(SMA_EYE_HEIGHTS, EYE_HEIGHTS, SMA_N)
            
            # Reshape lists
            EYE_HEIGHTS.limit()
            SMA_EYE_HEIGHTS.limit()

        # Detect bad posture
        if SMA_EYE_HEIGHTS and (SMA_EYE_HEIGHTS.max() - SMA_EYE_HEIGHTS[-1] > 500 * D / Z):
            res = messagebox.showinfo("BAD POSTURE!", "Sit up straight!\nCorrect your posture, then click ok.")
            if res == "ok":
                # Initialize state, and restart from begening
                EYE_HEIGHTS = LimitedList()
                SMA_EYE_HEIGHTS = LimitedList()
                graph_x = np.arange(0, PLOT_NUM*PLOT_DELTA, PLOT_DELTA)
                continue

        # Plot eye heights
        graph_x += PLOT_DELTA
        ax.set_xlim((graph_x.min(), graph_x.max()))
        ax.set_ylim(0, CAP_FRAME_HEIGHT)

        if len(EYE_HEIGHTS) >= PLOT_NUM:
            eye_y = EYE_HEIGHTS[-PLOT_NUM:]
            eye_lines.set_data(graph_x, eye_y)
            plt.pause(.001)
        
        if len(SMA_EYE_HEIGHTS) >= PLOT_NUM:
            smas_eye_y = SMA_EYE_HEIGHTS[-PLOT_NUM:]
            smas_eye_lines.set_data(graph_x, smas_eye_y)
            plt.pause(.001)

        
        # Show video
        cv2.imshow(WINDOW_NAME, img_gray)

        # Quit with ESC Key
        if cv2.waitKey(1) & 0xFF == 27:
            break
    
    # End processing
    cv2.destroyAllWindows()
    cap.release()
