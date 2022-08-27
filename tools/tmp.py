import csv
import re
import enchant
import pandas as pd

def restore_from_file(file_path):
  restored_lst = []
  with open(file_path, "r") as file:
    for line in file.readlines():
      if line != "":
        restored_lst.append(eval(line))
  return restored_lst

def data_to_vector(data):
    vec = []
    for frame in data:
        avg = [0, 0, 0, 0, 0, 0, 0]
        for face in frame:
            avg[0] += face["angry"]
            avg[1] += face["disgust"]
            avg[2] += face["fear"]
            avg[3] += face["happy"]
            avg[4] += face["sad"]
            avg[5] += face["surprise"]
            avg[6] += face["neutral"]
        if len(frame) != 0:
            for i in range(7):
                avg[i] /= len(frame)
            vec.append(avg)
    return vec

d = enchant.Dict("en_US")
def process_text(text):
    result = re.sub(r"[\n\"\[\]~;]", "", text)
    lst = result.split()
    s = ""
    for item in lst:
        item = item.strip()
        if (d.check(item) and len(item)!=1) or item == "a" or item == "I" or item == "i" or item == "A":
            s += " "+item
    #result = re.sub(r"\n", " [SEP] ", result)[:-7]
    if len(s)<6:
        s = ""
    return s

def get_text(path):
    try:
        f = open(path, "r")
        d = restore_from_file(path)
        vec = data_to_vector(d)
        if len(vec) == 0:
            return ""
        return str(vec)
    except OSError as e:
        print(str(e))
        return ""
        
        
'''def get_text(path):
    try:
        f = open(path, "r")
        res = ""
        for row in f.readlines():
            k = process_text(row.lower())
            if len(k) > 0:
                res += process_text(row.lower()) + ", "
        return res[:-2]
    except OSError:
        return ""'''

with open("training/data/tagging_MMD_db_with_sentiment.csv", "r") as csvinput:
    with open("training/data/tagging_new.csv", "w") as csvoutput:
        writer = csv.writer(csvoutput, lineterminator='\n')
        reader = csv.reader(csvinput)

        a = []
        row = next(reader)
        row.append("face_sentiment")
        a.append(row)

        for row in reader:
            name = row[1]
            txt = get_text("training/data/video_facial_data/"+name+"_data")
            row.append(txt)
            a.append(row)

        writer.writerows(a)
