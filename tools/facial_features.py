import os
import shutil
from retinaface import RetinaFace
from deepface import DeepFace
import json
from video_tools import generate_frames

FRAMES_PATH = "tmp_frames_faces"

def retrieve_faces_data(video_path, rate = 50, show_print = True):
    faces_lst = []
    generate_frames(video_path, FRAMES_PATH, rate = rate, show_print = show_print)
    for i in sorted([int(s[:-4]) for s in os.listdir(FRAMES_PATH)]):
      faces = RetinaFace.extract_faces(FRAMES_PATH + "/" + str(i) + ".png")
      data_lst = []
      for face in faces:
        try:
          face_dict = DeepFace.analyze(face, actions = ["emotion"], detector_backend = "skip")
          data_lst.append(face_dict["emotion"])
        except ValueError:
          # Face was not detected
          continue
      faces_lst.append(data_lst)
    # Delete temporary directory
    #shutil.rmtree(FRAMES_PATH)
    return faces_lst

def retrieve_to_file(dest, video_path):
  face_data = retrieve_faces_data(video_path, show_print = False)
  with open(dest, "w") as output_file:
        output_file.writelines([json.dumps(item) + "\n" for item in face_data])

def retrieve_to_files(dest, video_path):
  for file_name in os.listdir(video_path):
      retrieve_to_file(dest + "/" + os.path.splitext(file_name)[0] + "_data", video_path + "/" + file_name)

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

if __name__ == "__main__":
    retrieve_to_files("x", "result")