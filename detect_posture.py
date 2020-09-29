import sys
import numpy as np
from matplotlib import pyplot as plt
import cv2
import tkinter as tk
from tkinter import messagebox


WINDOW_NAME = "capture"     # Videcapute window name
CAP_FRAME_WIDTH = 640       # Videocapture width
CAP_FRAME_HEIGHT = 480      # Videocapture height
CAP_FRAME_FPS = 30          # Videocapture fps (depends on user camera)

DEVICE_ID = 1               # Web camera id (0 is maybe built-in camera)

SMA_SEC = 10                        # SMA seconds
SMA_N = SMA_SEC * CAP_FRAME_FPS     # SMA n

PLOT_NUM = 20                   # Plot points number
PLOT_DELTA = 1/CAP_FRAME_FPS    # Step of X axis

Z = 45                  # (cm) Distance from PC to face 
D = 3                   # (cm) Limit of lowering eyes
F = 500                 # Focal length


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


def simple_moving_average(n, data):
    """ Return simple moving average """
    result = []
    for m in range(n-1, len(data)):
        total = sum([data[m-i] for i in range(n)])
        result.append(total/n)
    return result

def add_simple_moving_average(smas, n, data):
    """ Add simple moving average """
    total = sum([data[-1-i] for i in range(n)])
    smas.append(total/n)


if __name__ == '__main__':
    # Not show tkinter window
    root = tk.Tk()
    root.iconify()

    # Count the number of cameras.
    # If there are no cameras available, end program.
    if not count_camera_connection():
        sys.exit(1)

    # Chose cascade
    cascade = cv2.CascadeClassifier("haarcascade_eye.xml")

    # Capture setup
    cap = cv2.VideoCapture(DEVICE_ID, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAP_FRAME_FPS)

    # Prepare windows
    cv2.namedWindow(WINDOW_NAME)

    # Time series data of eye height
    eye_heights = []
    sma_eye_heights = []

    # Plot setup
    ax = plt.subplot()
    graph_x = np.arange(0, PLOT_NUM*PLOT_DELTA, PLOT_DELTA)
    eye_y = [0] * PLOT_NUM
    sma_eye_y = [0] * PLOT_NUM
    eye_lines, = ax.plot(graph_x, eye_y, label="realtime")
    sma_eye_lines, = ax.plot(graph_x, sma_eye_y, label="SMA")
    ax.legend()


    while cap.isOpened():
        # Get a frame
        ret, frame = cap.read()

        # Convert image to gray scale
        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect human eyes
        eyes = cascade.detectMultiScale(img_gray, minSize=(30, 30))

        # Mark on the detected eyes
        for (x, y, w, h) in eyes:
            color = (255, 0, 0)
            cv2.rectangle(img_gray, (x, y), (x+w, y+h), color, thickness=3)
        
        # Store eye heights
        if len(eyes) > 0:
            eye_average_height = CAP_FRAME_HEIGHT - sum([y for _, y, _, _ in eyes]) / len(eyes)
            eye_heights.append(eye_average_height)

            if len(eye_heights) == SMA_N:
                sma_eye_heights = simple_moving_average(SMA_N, eye_heights)
            elif len(eye_heights) > SMA_N:
                add_simple_moving_average(sma_eye_heights, SMA_N, eye_heights)
            

        # Detect bad posture
        if sma_eye_heights and (sma_eye_heights[0] - sma_eye_heights[-1] > F * D / Z):
            res = messagebox.showinfo("BAD POSTURE!", "Sit up straight!\nCorrect your posture, then click ok.")
            if res == "ok":
                # Initialize state, and restart from begening
                eye_heights = []
                sma_eye_heights = []
                graph_x = np.arange(0, PLOT_NUM*PLOT_DELTA, PLOT_DELTA)
                continue

        # Plot eye heights
        graph_x += PLOT_DELTA
        ax.set_xlim((graph_x.min(), graph_x.max()))
        ax.set_ylim(0, CAP_FRAME_HEIGHT)

        if len(eye_heights) >= PLOT_NUM:
            eye_y = eye_heights[-PLOT_NUM:]
            eye_lines.set_data(graph_x, eye_y)
            plt.pause(.001)
        
        if len(sma_eye_heights) >= PLOT_NUM:
            sma_eye_y = sma_eye_heights[-PLOT_NUM:]
            sma_eye_lines.set_data(graph_x, sma_eye_y)
            plt.pause(.001)

        
        # Show result
        cv2.imshow(WINDOW_NAME, img_gray)

        # Quit with ESC Key
        if cv2.waitKey(1) & 0xFF == 27:
            break
    
    # End processing
    cv2.destroyAllWindows()
    cap.release()
