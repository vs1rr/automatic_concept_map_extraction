# -*- coding: utf-8 -*-
"""
Get final results
"""
import os
import json
import click
import numpy as np

def read_json(json_path):
    with open(json_path, "r", encoding="utf-8") as openfile:
        data = json.load(openfile)
    return data

def avg_results(metrics):
    res = {
        "meteor": {x: [] for x in ["precision", "recall", "f1"]},
        "rouge-2": {x: [] for x in ["precision", "recall", "f1"]}
    }
    for _, info in metrics.items():
        for k1, val in info.items():
            for k2, metric in val.items():
                res[k1][k2].append(metric)
    for k1, v in res.items():
        for k2, l in v.items():
            res[k1][k2] = np.mean(l)
    return res

@click.command()
@click.argument("exp_path")
def main(exp_path):
    exps = sorted([x for x in os.listdir(exp_path) if os.path.isdir(os.path.join(exp_path, x))])[:3]
    exps = [(x.split("-")[-1], read_json(os.path.join(exp_path, x, "metrics.json"))) for x in exps]
    exps = [(x, avg_results(metrics)) for (x, metrics) in exps]

    for (name, results) in exps:
        print(f"TYPE EXP: {name}")
        for k1, v in results.items():
            rounded = {k: np.round(score, 2) for k, score in v.items()}
            print(f"{k1}\t{rounded}")
        print("==========")


if __name__ == '__main__':
    main()
