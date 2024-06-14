# -*- coding: utf-8 -*-
"""

"""
import os
import pandas as pd

def main(df_input):
    nb = int(df_input.shape[0]//10)
    return df_input[:8*nb], df_input[8*nb:9*nb], df_input[9*nb:]


if __name__ == '__main__':
    FOLDER = "src/fine_tune_rebel/"
    DF_INPUT = pd.read_csv(os.path.join(FOLDER, "cm_biology.csv"))
    DF_INPUT = DF_INPUT[[col for col in DF_INPUT.columns if col != "Unnamed: 0"]]

    TRAIN, EVAL, TEST = main(df_input=DF_INPUT)
    TRAIN.to_csv(os.path.join(FOLDER, "cm_biology_train.csv"))
    EVAL.to_csv(os.path.join(FOLDER, "cm_biology_eval.csv"))
    TEST.to_csv(os.path.join(FOLDER, "cm_biology_test.csv"))
