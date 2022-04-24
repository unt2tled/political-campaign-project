"""
This module contains all the necessary functions for parsing training data
"""
import csv

TAGGING_PATH = "data/tagging.csv"
SUBSCRIPTS_FOLDER_PATH = "data/video_subsripts"


def string_from_transription(video_name: str) -> str:
    """
    Loads SUBSCRIPTS_FOLDER_PATH/[video_name]_text.txt file, removes all the metadata (timestamps, etc.), linebreaks
    symbols and returns transcription as a string object.

    :param video_name: name string of the video, For example: PRES_TRUSTEDLEADERSHIP_KASICH_WON'T_PLAY.
    """
    file = open(video_name, "r")
    data = file.readlines()[4::2]
    ret = ""
    for line in data:
        ret += line.strip()
        ret += " "
    return ret



def get_label_by_maj(labels_lst: list) -> int:
    """
    Receives a list of integer labels (-1=center, 1=base, 0=both) and returns the most frequent one.

    :param labels_lst: list of labels.
    """
    return max(set(labels_lst), key=labels_lst.count)


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


def build_csv_from_taggings(dest_path: str):
    """
    Builds a new .csv file with three columns "name", "subscript" and "label" from tagging and subscripts files
    specified in TAGGING_PATH and SUBSCRIPTS_FOLDER_PATH variables.
    The "name" column contains names of the videos in upper case, as it appears in TAGGING_PATH file.
    The "subscript" column contains subscripts of the videos.
    The "label" column contains final labels of the videos returned by get_label_by_maj function.
    Possible labels are: -1=center, 1=base, 0=both.

    :param dest_path: destination path of a newly created .csv file.
    """
    with open(TAGGING_PATH, "r") as tagging_file:
        tagging_reader = csv.DictReader(tagging_file)
        with open(dest_path, "w", newline="") as output_file:
            fieldnames = ["text", "label"]
            writer = csv.DictWriter(output_file, fieldnames=fieldnames)
            writer.writeheader()
            for row in tagging_reader:
                title = row["Title"]
                tagging_list = retrieve_tags(row.values())
                # Ignore empty labeling
                if not tagging_list:
                    continue
                # Calculate label
                label = get_label_by_maj(tagging_list)
                writer.writerow({"text": string_from_transription(title), "label": label})


str = string_from_transription("C:/Users/maor9/IdeaProjects/political-campaign-project/text files/PRES-45COMMITTEE-50-POINTS-AHEAD (English)_text.txt")
