"""
This module contains all the necessary functions for parsing training data
"""

TAGGINGS_PATH = "data/tagging.csv"
SUBSCRIPTS_FOLDER_PATH = "data/video_subsripts"

def string_from_transription(video_name: str) -> str:
    """
    Loads SUBSCRIPTS_FOLDER_PATH/[video_name]_text.txt file, removes all the metadata (timestamps, etc.), linebreaks
    symbols and returns transcription as a string object.

    :param video_name: name string of the video, For example: PRES_TRUSTEDLEADERSHIP_KASICH_WON'T_PLAY.
    """
    pass

def get_label_by_maj(labels_lst: list) -> int:
    """
    Receives a list of integer labels (-1=center, 1=base, 0=both) and returns the most frequent one.

    :param labels_lst: list of labels.
    """
    return max(set(labels_lst), key=labels_lst.count)

def build_csv_from_taggings(dest_path: str):
    """
    Builds a new .csv file with three columns "name", "subscript" and "label" from taggings and subscripts files
    specified in TAGGINGS_PATH and SUBSCRIPTS_FOLDER_PATH variables.
    The "name" column contains names of the videos in upper case, as it appears in TAGGINGS_PATH file.
    The "subscript" column contains subscripts of the videos.
    The "label" column contains final labels of the videos returned by get_label_by_maj function.
    Possible labels are: -1=center, 1=base, 0=both.

    :param dest_path: destination path of a newly created .csv file.
    """
    pass
