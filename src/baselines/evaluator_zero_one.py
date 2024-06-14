import os
import json
from datetime import datetime
from typing import List
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

class EvaluationMetrics:
    """Evaluation metrics for each file"""

    def __init__(self):
        self.meteor = meteor_score
        self.rouge_metrics = ['rouge1', 'rouge2']
        self.rouge = rouge_scorer.RougeScorer(self.rouge_metrics, use_stemmer=True)

    @staticmethod
    def get_rouge_input(triples_input):
        triples_input = [" ".join(x) for x in triples_input]
        return " . ".join(triples_input)

    def __call__(self, triples: List, gold_triples: List):
        """
        Adapted ROUGE/METEOR, as in Falke's

        nb of rows = nb of `triples`
        nb of columns = nb of `gold_triples`
        """
        nb_t, nb_gt = len(triples), len(gold_triples)
        meteor_cached_recall = np.zeros((nb_t, nb_gt))
        meteor_cached_precision = np.zeros((nb_t, nb_gt))

        rouge_t = self.get_rouge_input(triples_input=triples)
        rouge_t_gold = self.get_rouge_input(triples_input=gold_triples)
        scores = self.rouge.score(rouge_t_gold, rouge_t)

        for i, t_i in enumerate(triples):
            for j, t_j in enumerate(gold_triples):
                meteor_t = word_tokenize(" ".join(t_i))
                meteor_t_gold = word_tokenize(" ".join(t_j))
                meteor_cached_recall[i][j] = self.meteor([meteor_t], meteor_t_gold)
                meteor_cached_precision[i][j] = self.meteor([meteor_t_gold], meteor_t)

        meteor_r = np.sum(np.max(meteor_cached_recall, axis=1)) / nb_t
        meteor_p = np.sum(np.max(meteor_cached_recall, axis=0)) / nb_gt

        return {
            "meteor": {
                "precision": 100 * meteor_p,
                "recall": 100 * meteor_r,
                "f1": 100 * 2 * meteor_p * meteor_r / (meteor_p + meteor_r)},
            "rouge-2": {
                "precision": 100 * scores["rouge2"].precision,
                "recall": 100 * scores["rouge2"].recall,
                "f1": 100 * scores["rouge2"].fmeasure}
        }

def read_triples(file_path):
    with open(file_path, encoding="utf-8") as f:
        lines = f.readlines()
    triples = list(set(lines))
    triples = [x.strip().split(", ") for x in triples]
    return triples

def read_gold_triples(file_path):
    with open(file_path, encoding="utf-8") as f:
        lines = f.readlines()
    gold_triples = [x.strip().split("\t") for x in lines]
    return gold_triples

if __name__ == '__main__':
    base_path = "./experiments_emnlp/baselines/output_zero_one_baseline/one-train"
    gold_base_path = "./data/Corpora_Falke/Wiki/train"
    save_folder ="./experiments_emnlp/baselines/one_baseline/results-one-train"
    os.makedirs(save_folder, exist_ok=True)

    logs = {}

    METRICS = EvaluationMetrics()

    for file_name in os.listdir(base_path):
        if file_name.endswith(".csv"):
            concept_map_number = file_name.split("_")[-1].split(".")[0]
            triples_file_path = os.path.join(base_path, file_name)
            gold_triples_file_path = os.path.join(gold_base_path, concept_map_number, f"{concept_map_number}.cmap")

            if os.path.exists(gold_triples_file_path):
                start_ = datetime.now()
                print(f"Processing {file_name} and {concept_map_number}.cmap")

                TRIPLES = read_triples(triples_file_path)
                GOLD_TRIPLES = read_gold_triples(gold_triples_file_path)

                RES = METRICS(triples=TRIPLES, gold_triples=GOLD_TRIPLES)

                # Save metrics
                metrics_save_path = os.path.join(save_folder, f"metrics_{concept_map_number}.json")
                with open(metrics_save_path, "w", encoding="utf-8") as openfile:
                    json.dump(RES, openfile, indent=4)

                end_ = datetime.now()

                # Update logs
                if concept_map_number not in logs:
                    logs[concept_map_number] = {}
                logs[concept_map_number][file_name] = {
                    "start": str(start_),
                    "end": str(end_),
                    "total": str(end_ - start_)
                }

                # Save logs
                logs_save_path = os.path.join(save_folder, "logs.json")
                with open(logs_save_path, "w", encoding="utf-8") as openfile:
                    json.dump(logs, openfile, indent=4)

                print(f"Total execution time: {(end_ - start_).total_seconds():.4f}s")
            else:
                print(f"Gold file {concept_map_number}.cmap does not exist")
