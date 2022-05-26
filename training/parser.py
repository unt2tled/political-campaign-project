"""
This module contains all the necessary functions for parsing training data
"""
import csv
import re

TAGGING_PATH = "data/tagging.csv"
SUBSCRIPTS_FOLDER_PATH = "data/video_subscripts"


def string_from_transription(video_name: str, remove_char="\"") -> str:
    """
    Loads SUBSCRIPTS_FOLDER_PATH/[video_name]_text.txt file, removes all the metadata (timestamps, etc.), linebreaks
    symbols and returns transcription as a string object.

    :param video_name: name string of the video, For example: PRES_TRUSTEDLEADERSHIP_KASICH_WON'T_PLAY.
    :param remove_char charaters to remove from the returned string
    """
    try:
        file = open(SUBSCRIPTS_FOLDER_PATH + "/" + video_name + "_text.txt", "r")
    except FileNotFoundError:
        return ""
    data = file.readlines()
    ret = ""
    for line in data:
        ret += line.strip()
        ret += " "
    return ret.translate({ord(k):None for k in remove_char})


def get_label_by_maj(labels_lst: list, label_type: str) -> int:
    """
    Receives a list of integer labels (-1=center, 1=base, 0=both) and returns tag 0 or 1 for base or center by the most frequent one.

    :param labels_lst: list of labels.
    :param label_type: type of labeel (base or center).
    """
    s = sum(labels_lst)
    label_num = 0;
    if label_type == "base":
        label_num = 1
    if s > 0:
        return label_num
    elif s < 0:
        return 1-label_num
    return 1


def retrieve_tags(tags_lst: list) -> list:
    """
    Accepts list of strings and returns list of integers such that every label substituted with its numerical value
    (-1=center, 1=base, 0=both)
    :param tags_lst:
    """
    result_lst = []
    for entry in tags_lst:
        if entry == "base":
            result_lst.append(1)
        elif entry == "center":
            result_lst.append(-1)
        elif entry == "both":
            result_lst.append(0)
    return result_lst


def build_csv_from_taggings(dest_path: str, label_type: str, remove_char="\""):
    """
    Builds a new .csv file with three columns "name", "subscript" and "label" from tagging and subscripts files
    specified in TAGGING_PATH and SUBSCRIPTS_FOLDER_PATH variables.
    The "name" column contains names of the videos in upper case, as it appears in TAGGING_PATH file.
    The "subscript" column contains subscripts of the videos.
    The "label" column contains final labels of the videos.
    
    :param dest_path: destination path of a newly created .csv file.
    """
    with open(TAGGING_PATH, "r") as tagging_file:
        tagging_reader = csv.DictReader(tagging_file)
        with open(dest_path, "w", newline="") as output_file:
            fieldnames = ["name", "text", "label"]
            writer = csv.DictWriter(output_file, fieldnames=fieldnames)
            writer.writeheader()
            for row in tagging_reader:
                title = row["Title"]
                tagging_list = retrieve_tags(row.values())
                # Ignore empty labeling
                if not tagging_list:
                    continue
                # Calculate label
                label = get_label_by_maj(tagging_list, label_type)
                transcription = string_from_transription(title, remove_char=remove_char)
                # Ignore empty trascripts
                if transcription == "":
                    continue
                transcription = re.sub("\d\d:\d\d:\d\d Speaker 1","",transcription)
                writer.writerow({"name": title, "text": transcription, "label": label})


if __name__ == "__main__":
    build_csv_from_taggings("tags_base.csv", "base")
    build_csv_from_taggings("tags_center.csv", "center")
