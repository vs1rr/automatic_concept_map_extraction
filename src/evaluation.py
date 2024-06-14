"""
Evaluation metrics
"""
from typing import List
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer


class EvaluationMetrics:
    """ evaluation """

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

        if nb_t == 0:
            return {
                "meteor": {"precision": 0, "recall": 0, "f1": 0},
                "rouge-2": {"precision": 0, "recall": 0, "f1": 0}
            }
        # meteor[i][j] -> meteor(i, j)
        meteor_cached_recall = np.zeros((nb_t, nb_gt))
        # meteor[i][j] -> meteor(j, i)
        meteor_cached_precision = np.zeros((nb_t, nb_gt))

        # ROUGE
        rouge_t = self.get_rouge_input(triples_input=triples)
        rouge_t_gold = self.get_rouge_input(triples_input=gold_triples)
        # scores = self.rouge.score(rouge_t, rouge_t_gold)
        scores = self.rouge.score(rouge_t_gold, rouge_t)

        # Meteor
        for i, t_i in enumerate(triples):
            for j, t_j in enumerate(gold_triples):
                meteor_t = word_tokenize(" ".join(t_i))
                meteor_t_gold = word_tokenize(" ".join(t_j))
                meteor_cached_recall[i][j] = self.meteor([meteor_t], meteor_t_gold)
                meteor_cached_precision[i][j] = self.meteor([meteor_t_gold], meteor_t)

        meteor_r = np.sum(np.max(meteor_cached_recall, axis=1)) / nb_t
        meteor_p = np.sum(np.max(meteor_cached_precision, axis=0)) / nb_gt

        return {
            "meteor": {
                "precision": 100 * meteor_p,
                "recall": 100 * meteor_r,
                "f1": 100 * 2 * meteor_p * meteor_r / (meteor_p + meteor_r) if (meteor_p + meteor_r) else 0},
            "rouge-2": {
                "precision": 100 * scores["rouge2"].precision,
                "recall": 100 * scores["rouge2"].recall,
                "f1": 100 * scores["rouge2"].fmeasure}
        }


if __name__ == '__main__':
    TRIPLES = open("experiments/2024-01-26-12:18:28/data_test/relation/M1.txt", encoding="utf-8").readlines()
    TRIPLES = list(set(TRIPLES))
    TRIPLES = [x.replace("\n", "").split(", ") for x in TRIPLES]
    print(len(TRIPLES))

    GOLD_TRIPLES = open("./data_test/101.cmap", encoding="utf-8").readlines()
    GOLD_TRIPLES = [x.replace("\n", "").split("\t") for x in GOLD_TRIPLES]
    print(len(GOLD_TRIPLES))

    METRICS = EvaluationMetrics()
    RES = METRICS(triples=TRIPLES, gold_triples=GOLD_TRIPLES)
    print(RES)