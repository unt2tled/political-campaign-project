import easyocr
import os
import cv2
import shutil
import difflib

FRAMES_PATH = "tmp_frames"
CONF_THRESH = 0.6
SIMILARITY_THRESH = 0.8

def generate_frames(video_path, rate, show_print = True):
    # Create a new temporary folder
    if not os.path.exists(FRAMES_PATH):
        os.makedirs(FRAMES_PATH)
    # Capture video
    src_vid = cv2.VideoCapture(video_path)
    index = 0
    while src_vid.isOpened():
        ret, frame = src_vid.read()
        if not ret:
            break
        name = FRAMES_PATH + "/frame" + str(index) + ".png"
        if index % rate == 0:
            if show_print:
                print("Frame: " + name)
            cv2.imwrite(name, frame)
        index = index + 1
    src_vid.release()

def add_text(text_lst, text):
    for t in text_lst:
      similarity = difflib.SequenceMatcher(None, t, text).ratio()
      if similarity > SIMILARITY_THRESH:
          return
    text_lst.append(text)

def retrieve_text(video_path, rate=5):
    texts_lst = []
    generate_frames(video_path, rate = rate)
    ocr = easyocr.Reader(['en'])
    for i in os.listdir(FRAMES_PATH):
        text = ocr.readtext(FRAMES_PATH + "/" + i)
        for txt in text:
          # Threshold for confidence
          if txt[2] > CONF_THRESH:
            # Filter similar texts
            add_text(texts_lst, txt[1])
    # Delete temporary directory
    shutil.rmtree(FRAMES_PATH)
    return texts_lst
