import easyocr
import os
import cv2
import shutil
import difflib
from tools.video_tools import generate_frames

FRAMES_PATH = "tmp_frames"
CONF_THRESH = 0.9
SIMILARITY_THRESH = 0.8


def add_text(text_lst, text):
    for t in text_lst:
      similarity = difflib.SequenceMatcher(None, t, text).ratio()
      if similarity > SIMILARITY_THRESH:
          return
    text_lst.append(text)

def retrieve_text(video_path, rate = 5, show_print = True):
    texts_lst = []
    generate_frames(video_path, FRAMES_PATH, rate = rate, show_print = show_print)
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

def retrieve_to_file(dest, video_path):
  text_lst = retrieve_text(video_path, rate = 2, show_print = False)
  file = open(dest, "w")
  file.writelines([line + "\n" for line in text_lst])
  file.close()

def retrieve_to_files(dest, video_path):
  for file_name in os.listdir(video_path):
      retrieve_to_file(dest + "/" + os.path.splitext(file_name)[0] + "_text.txt", video_path + "/" + file_name)
