import os
import cv2

def generate_frames(video_path, frames_path, rate, show_print = True):
    # Create a new temporary folder
    if not os.path.exists(frames_path):
        os.makedirs(frames_path)
    # Capture video
    src_vid = cv2.VideoCapture(video_path)
    index = 0
    while src_vid.isOpened():
        ret, frame = src_vid.read()
        if not ret:
            break
        name = frames_path + "/" + str(index) + ".png"
        if index % rate == 0:
            if show_print:
                print("Frame: " + name)
            cv2.imwrite(name, frame)
        index = index + 1
    src_vid.release()
