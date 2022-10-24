from tempfile import NamedTemporaryFile
import shutil
import csv
from colors_feature import retrieve_colors_from_video, retrieve_colors_per_frame
import os

VIDEOS_PATH = "C:\\Users\\un.t1tled\\Desktop\\videos\\"

def filter_videos(taggings_path):
    with open(taggings_path, "r") as csvfile:
        reader = csv.DictReader(csvfile)
        folder_name = "videos"
        os.mkdir(folder_name)
        for row in reader:
            name = row["name"]
            path = VIDEOS_PATH+name+".wmv"
            try:
                shutil.copyfile(path, folder_name+"/"+name+".wmv")
            except FileNotFoundError:
                print(name+" is not found.")

def fill_features(filename):
    temp_file = NamedTemporaryFile(mode="w", delete=False, newline="")
    with open(filename, "r") as csvfile, temp_file:
        reader = csv.DictReader(csvfile)
        fields = reader.fieldnames
        writer = csv.DictWriter(temp_file, fieldnames=fields)
        writer.writeheader()
        counter = 0
        for row in reader:
            counter += 1
            print(counter)
            name = row["name"]
            path = VIDEOS_PATH+name+".wmv"
            row["color_total"] = str(retrieve_colors_from_video(path, show_print = False))
            row["color_tails"] = str(retrieve_colors_from_video(path, threshold = slice(1, 3), show_print = False))+"\n"+str(retrieve_colors_from_video(path, threshold = slice(None, -4, None), show_print = False))
            writer.writerow(row)
    shutil.move(temp_file.name, filename)

if __name__ == "__main__":
    filter_videos("tagging_db.csv")
