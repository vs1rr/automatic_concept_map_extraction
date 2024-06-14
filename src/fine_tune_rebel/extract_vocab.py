# -*- coding: utf-8 -*-
"""

"""
import pandas as pd

def get_vocab(df_input):
    res = df_input.triplets.values
    res = [x.split("<obj> ")[1] for x in res]
    return list(set(res))

if __name__ == '__main__':
    DF_INPUT = pd.read_csv("src/fine_tune_rebel/cm_biology_train.csv")
    VOCAB = get_vocab(df_input=DF_INPUT)
    print(VOCAB)