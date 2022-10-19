# ref: https://towardsdatascience.com/building-an-image-color-analyzer-using-python-12de6b0acf74
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np
import cv2
import shutil

FRAMES_PATH = "tmp_frames_colors"

def rgb_to_hex(rgb_color):
    hex_color = "#"
    for i in rgb_color:
        i = int(i)
        hex_color += ("{:02x}".format(i))
    return hex_color

def prep_image(raw_img):
    modified_img = cv2.resize(raw_img, (900, 600), interpolation = cv2.INTER_AREA)
    modified_img = modified_img.reshape(modified_img.shape[0]*modified_img.shape[1], 3)
    return modified_img

def color_analysis(img, show_diagram = False):
    clf = KMeans(n_clusters = 8, n_init = 50, max_iter = 500)
    color_labels = clf.fit_predict(img)
    center_colors = clf.cluster_centers_
    counts = Counter(color_labels)
    total = sum([counts[i] for i in counts.keys()])
    ordered_colors = [center_colors[i] for i in counts.keys()]
    hex_colors = [rgb_to_hex(ordered_colors[i]) for i in counts.keys()]
    colors_stat = [(rgb_to_hex(ordered_colors[i]), counts[i]/total) for i in counts.keys()]
    if show_diagram:
      plt.figure(figsize = (12, 8))
      plt.pie(counts.values(), labels = hex_colors, colors = hex_colors)
      plt.show()
    return colors_stat
    
def retrieve_colors_per_frame(video_path, rate = 10, show_print = True):
    generate_frames(video_path, FRAMES_PATH, rate = rate, show_print = show_print)
    for i in sorted([int(s[:-4]) for s in os.listdir(FRAMES_PATH)]):
      image = cv2.imread(FRAMES_PATH + "/" + str(i) + ".png")
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      modified_image = prep_image(image)
      color_analysis(modified_image)
    # Delete temporary directory
    shutil.rmtree(FRAMES_PATH)

def retrieve_colors_from_video(video_path, rate = 5, show_print = True, threshold = slice(None, -1, None)):
    generate_frames(video_path, FRAMES_PATH, rate = rate, show_print = show_print)
    data_points = np.empty(shape = (0, 3))
    for i in sorted([int(s[:-4]) for s in os.listdir(FRAMES_PATH)])[threshold]:
      image = cv2.imread(FRAMES_PATH + "/" + str(i) + ".png")
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      modified_image = prep_image(image)
      data_points = np.vstack((data_points, modified_image))
    # Delete temporary directory
    shutil.rmtree(FRAMES_PATH)
    return color_analysis(modified_image)
