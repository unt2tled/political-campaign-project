from __future__ import print_function

import os
import subprocess

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
import graphviz

# ref: http://chrisstrelioff.ws/sandbox/2015/06/08/decision_trees_in_python_with_scikit_learn_and_pandas.html

input_file_path = 'text_words_labels.csv'

def get_data(input_file_path):
    df = pd.read_csv(input_file_path)
    return df

def encode_target(df, target_column):
    """Add column to df with integers for the target.

    Args
    ----
    df -- pandas DataFrame.
    target_column -- column to map to int, producing
                     new Target column.

    Returns
    -------
    df_mod -- modified DataFrame.
    targets -- list of target names.
    """
    df_mod = df.copy()
    targets = df_mod[target_column].unique()
    map_to_int = {name: n for n, name in enumerate(targets)}
    df_mod["target"] = df_mod[target_column].replace(map_to_int)

    return (df_mod, targets)

df = get_data(input_file_path)
df2, targets = encode_target(df, "target")
print("* df2.head()", df2[["target", "name"]].head(),
      sep="\n", end="\n\n")
print("* df2.tail()", df2[["target", "name"]].tail(),
      sep="\n", end="\n\n")
print("* targets", targets, sep="\n", end="\n\n")

features = [c for c in df2.columns.values if c != 'name' and c != 'isdefinite' and c != 'target']

y = df2["target"]
X = df2[features]
dt = DecisionTreeClassifier(min_samples_split=20, random_state=99)
dt.fit(X, y)

plot_tree(dt,max_depth=3)
