from tempfile import NamedTemporaryFile
import shutil
import csv
from colors_feature import retrieve_colors_from_video, retrieve_colors_per_frame

filename = "tagging_db1.csv"
tempfile = NamedTemporaryFile(mode="w", delete=False, newline="")

with open(filename, "r") as csvfile, tempfile:
    reader = csv.DictReader(csvfile)
    fields = reader.fieldnames
    writer = csv.DictWriter(tempfile, fieldnames=fields)
    writer.writeheader()
    counter = 0
    for row in reader:
        counter += 1
        print(counter)
        name = row["name"]
        path = "C:\\Users\\un.t1tled\\Desktop\\videos\\"+name+".wmv"
        #row["color_total"] = str(retrieve_colors_from_video(path, show_print = False))
        row["color_tails"] = str(retrieve_colors_from_video(path, threshold = slice(1, 3), show_print = False))+"\n"+str(retrieve_colors_from_video(path, threshold = slice(None, -4, None), show_print = False))
        writer.writerow(row)

shutil.move(tempfile.name, filename)
