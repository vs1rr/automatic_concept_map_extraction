# -*- coding: utf-8 -*-
"""
Get aggregated results from all experiments
"""
import os
import json
import scipy
import numpy as np
import pandas as pd
from loguru import logger
from src.build_table import build_table
import warnings

####### PARAMS BELOW TO UPDATE
SAVE_FOLDER = "./experiments"
DATA_PATH = "./Corpora_Falke/Wiki/train"
FOLDERS_CMAP = [x for x in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, x))]
DATE_START = "2024-06-04-11:00:00"
####################

COLUMNS = [
    'summary_method', 'summary_percentage',
    'ranking', 'ranking_how', 'ranking_perc_threshold',
    'entity', 'relation',
    'meteor_pr', 'meteor_re', 'meteor_f1',
    'rouge-2_pr', 'rouge-2_re', 'rouge-2_f1'
]


def read_json(json_path):
    with open(json_path, "r", encoding="utf-8") as openfile:
        data = json.load(openfile)
    return data


def avg_results(metrics):
    res = {
        x + "_" + y: [] for x in ["meteor", "rouge-2"] for y in ["pr", "re", "f1"]
    }

    for _, info in metrics.items():
        for k1, val in info.items():
            for k2, metric in val.items():
                res[f"{k1}_{k2[:2]}"].append(metric)

    for k1, v in res.items():
        res[k1] = round(np.mean(v), 1)

    return res


def get_folders_exp_finished():
    exps = os.listdir(SAVE_FOLDER)
    exps = [x for x in exps if os.path.isdir(os.path.join(SAVE_FOLDER, x))]
    valid_exps = []

    for x in exps:
        try:
            if all(y in os.listdir(os.path.join(SAVE_FOLDER, x)) \
                   for y in ["metrics.json", "params.json", "logs.json"] + FOLDERS_CMAP) and x >= DATE_START:
                valid_exps.append((read_json(os.path.join(SAVE_FOLDER, x, "logs.json")),
                                   read_json(os.path.join(SAVE_FOLDER, x, "params.json")),
                                   read_json(os.path.join(SAVE_FOLDER, x, "metrics.json"))))
        except NotADirectoryError:
            logger.warning(f"Skipped non-directory file: {x}")
        except Exception as e:
            logger.error(f"Error processing folder {x}: {e}")

    valid_exps = [x for x in valid_exps if x[0].get("finished") == "yes"]
    return [x[1:] for x in valid_exps]


def get_rebel_opt(params):
    options_rel = params["relation"]["options_rel"]
    local_rm = params["relation"]["local_rm"]
    if "rebel" in options_rel:
        x1 = "rebel\\_ft" if local_rm else "rebel\\_hf"
        x2 = "+dependency" if "dependency" in options_rel else ""
        return x1 + x2
    return "+".join(options_rel)


def format_vals(input, val1):
    input = [0 if x is None or (isinstance(x, float) and np.isnan(x)) else x for x in input]
    return [1 if x == val1 else 0 for x in input]

import pandas as pd
import scipy.stats


def get_correlations(df_, feat_cols, metric_cols):
    # Mapping to use in the correlations
    mappings = {
        "summary_method": {1: "chat-gpt", 2: "lex-rank"},
        "ranking_how": {1: "all", 2: "single"},
        "ranking": {1: "word2vec", 2: "page_rank"},
        # "entity": {1: "dbpedia_spotlight", 2: "nps"}
    }

    cols_df_corr_binary = ["Feature", "Val1", "Val2", "Metric", "Correlation", "Pvalue"]
    df_corr_binary = pd.DataFrame(columns=cols_df_corr_binary)

    constant_columns = []

    # Loop through mappings to calculate correlations
    for x, info in mappings.items():
        for metric in ["meteor_f1", "rouge-2_f1"]:
            # Handle cases where the column values might be None
            if None in df_[x]:
                continue
            vals_1 = format_vals(df_[x], info[1])
            vals_2 = df_[metric]

            # Catch ConstantInputWarning and print the constant column
            with warnings.catch_warnings():
                warnings.filterwarnings('error')
                try:
                    corr, pvalue = scipy.stats.spearmanr(vals_1, vals_2)
                    curr_l = [x.replace("_", "\\_"),
                              info[1], info[2], metric.replace("_", "\\_"),
                              round(corr, 3), "{:.2e}".format(pvalue)]
                    df_corr_binary.loc[len(df_corr_binary)] = curr_l
                except scipy.stats.ConstantInputWarning:
                    constant_columns.append((x, metric))

    print("Constant Columns:")
    for column in constant_columns:
        print(column)

    # Create LaTeX table for the correlations
    latex_table = build_table(
        columns=["Feature", "Metric", "Values", "Correlation", "Pvalue"],
        alignment="r" * len(cols_df_corr_binary),
        caption="Correlation between comparison of features and F1 scores",
        label="tab:wiki-train-binary-feat-corr-pval",
        position="h",
        data=df_corr_binary.values,
        sub_columns=["", "", "Val1", "Val2", "", ""],
        multicol=[1, 1, 2, 1, 1],
        resize_col=2
    )
    print(f"{latex_table}\n=====")

    relations = [
        ["rebel\_ft", "rebel\_hf"],
        ["rebel\_ft", "corenlp"],
        ["rebel\_hf", "corenlp"]]
    for [r_a, r_b] in relations:
        for metric in ["meteor_f1", "rouge-2_f1"]:
            # Handle cases where the column values might be None
            if None in df_["ranking"]:
                continue
            curr_df = df_[df_.ranking.isin([r_a, r_b])]
            vals_1 = format_vals(curr_df.ranking, r_a)
            vals_2 = curr_df[metric]
            corr, pvalue = scipy.stats.spearmanr(vals_1, vals_2)
            print(f"FEATURE: ranking | METRIC {metric} | VAL1 {r_a} | VAL2 {r_b}")
            print(f"Corr: {corr} | Pvalue: {pvalue}")
    print("==========")
    for r_a in ["rebel\_ft", "rebel\_hf", "corenlp"]:
        for metric in ["meteor_f1", "rouge-2_f1"]:
            # Handle cases where the column values might be None
            if None in df_["ranking"]:
                continue
            print(f"FEATURE: ranking | METRIC {metric} | VAL1 {r_a} | VAL2 Other")
            vals_1 = format_vals(df_.ranking, r_a)
            vals_2 = df_[metric]
            corr, pvalue = scipy.stats.spearmanr(vals_1, vals_2)
            print(f"Corr: {corr} | Pvalue: {pvalue}")


def f1_helper(row):
    if np.isnan(row["meteor_f1"]):
        return 2 * row["meteor_re"] * row["meteor_pr"] / (row["meteor_re"] + row["meteor_pr"])
    return row["meteor_f1"]


def main():
    df_output = pd.DataFrame(columns=COLUMNS)
    folders_exp = get_folders_exp_finished()
    logger.info(f"Results on {len(folders_exp)} experiments")

    for params, metrics in folders_exp:
        avg_metrics = avg_results(metrics)
        #start code changed to handle none value
        ranking_value = params["ranking"]["ranking"] if params["ranking"]["ranking"] is not None else "None"
        threshold_value = params["ranking"]["ranking_perc_threshold"] * 100 if params["ranking"][
                                                                                   "ranking_perc_threshold"] is not None else "None"

        summary_method = params["summary"]["summary_method"].replace("_", "\\_") if params["summary"][
                                                                                        "summary_method"] is not None else "None"

        curr_l = [
                     summary_method,
                     params["summary"]["summary_percentage"],
                     ranking_value.replace("_", "\\_"),
                     params["ranking"]["ranking_how"],
                     threshold_value,
                     "+".join(params["entities"]["options_ent"]).replace("_", "\\_"),
                     get_rebel_opt(params)
                 ] + [avg_metrics[f"{x}_{y}"] for x in ["meteor", "rouge-2"] for y in ["pr", "re", "f1"]]

        df_output.loc[len(df_output)] = curr_l
        #end code changed to handle none value

    print(df_output)
    df_output.meteor_f1 = df_output.apply(f1_helper, axis=1)
    df_output.sort_values(by=COLUMNS[:6]).to_csv("./pipeline/hp_search_results.csv")

    # latex_table = build_table(
    #     columns=["Summary", "Ranking", "Entity", "Relation", "METEOR", "ROUGE-2"],
    #     alignment="r"*len(COLUMNS),
    #     caption="Results for all systems on Wiki TRAIN",
    #     label="res-wiki-train-all-hyperparams",
    #     position="h",
    #     data=df_output.sort_values(by=COLUMNS[:7]).values,
    #     sub_columns=[x.replace("_", "\\_") for x in COLUMNS[:7]] + ["Pr", "Re", "F1"]*2,
    #     multicol=[2, 3, 1, 1, 3, 3],
    #     resize_col=2
    # )
    # print(latex_table)

    get_correlations(df_=df_output, feat_cols=COLUMNS[:6], metric_cols=["meteor_f1", "rouge-2_f1"])


if __name__ == '__main__':
    main()