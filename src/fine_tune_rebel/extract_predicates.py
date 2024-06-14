# -*- coding: utf-8 -*-
"""

"""
import os
from collections import Counter
import pandas as pd

def get_unique(folder):
    files = os.listdir(folder)
    dfs = []
    for file_x in files:
        try:
            dfs.append(pd.read_csv(os.path.join(folder, file_x), header=None,sep="\t"))
        except:
            print(f"File error for {file_x}")

    preds = [list(df[1].unique()) for df in dfs]
    # print(preds)
    # print(len(preds))

    pred_distrib = Counter([pred for l in preds for pred in l])
    return pred_distrib


if __name__ == '__main__':
    FOLDER = "src/fine_tune_rebel/all_gs_single"
    pred_distrib = get_unique(folder=FOLDER)
    print(pred_distrib)

    f = open("misc/predicate_label.txt", "w+", encoding="utf-8")
    for text in pred_distrib:
        f.write(f"{text}\n")
    f.close()
