import easyocr
import os
import cv2
import shutil
import difflib
import enchant
import re
from tools.video_tools import generate_frames

CONF_THRESH = 0.9
SIMILARITY_THRESH = 0.8

d = enchant.Dict("en_US")
def process_text(text):
    result = re.sub(r"[\n\"\[\]~;]", "", text)
    lst = result.split()
    s = ""
    for item in lst:
        item = item.strip()
        if (d.check(item) and len(item)!=1) or item == "a" or item == "I" or item == "i" or item == "A":
            s += " "+item
    if len(s)<6:
        s = ""
    return s

def get_formated_text(texts_arr):
    res = ""
    for row in texts_arr:
        k = process_text(row.lower())
        if len(k) > 0:
            res += process_text(row.lower()) + ", "
    return res[:-2]

def add_text(text_lst, text):
    for t in text_lst:
      similarity = difflib.SequenceMatcher(None, t, text).ratio()
      if similarity > SIMILARITY_THRESH:
          return
    text_lst.append(text)

def retrieve_text(video_path, rate = 5, frames_path = "tmp_frames", show_print = True):
    texts_lst = []
    generate_frames(video_path, frames_path, rate = rate, show_print = show_print)
    ocr = easyocr.Reader(['en'])
    for i in os.listdir(frames_path):
        text = ocr.readtext(frames_path + "/" + i)
        for txt in text:
          # Threshold for confidence
          if txt[2] > CONF_THRESH:
            # Filter similar texts
            add_text(texts_lst, txt[1])
    # Delete temporary directory
    shutil.rmtree(frames_path)
    return texts_lst

def retrieve_to_file(dest, video_path):
  text_lst = retrieve_text(video_path, rate = 2, show_print = False)
  file = open(dest, "w")
  file.writelines([line + "\n" for line in text_lst])
  file.close()

def retrieve_to_files(dest, video_path):
  for file_name in os.listdir(video_path):
      retrieve_to_file(dest + "/" + os.path.splitext(file_name)[0] + "_text.txt", video_path + "/" + file_name)
