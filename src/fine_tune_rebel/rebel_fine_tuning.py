# -*- coding: utf-8 -*-
"""
From: https://github.dev/ANR-kFLOW/KG2Narrative
"""
import sys
import pandas as pd
import nltk
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM  
import torch
from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm
import numpy as np

nltk.download('punkt')

LEARNING_RATE = 0.000025
EPOCHS = 10
BATCH_SIZE = 4
SEED = 1
SAVE_PATH = 'src/fine_tune_rebel/finetuned_rebel.pth'

STATS_REPORTS = {
    'best_metric': None, 'epoch': None, 'loss': None, 'val_loss': None
}


class DataSequence(torch.utils.data.Dataset):
    """ Dataset suitable for this finetuning"""
    def __init__(self, df_input):
        self.texts = tokenizer(
            df_input['context'].tolist(), padding='max_length', max_length=128,
            truncation=True, return_tensors="pt")

        self.labels = tokenizer(
            df_input['triplets'].to_list(), padding='max_length', max_length=128,
            truncation=True, return_tensors="pt")

    def __len__(self):
        return len(self.labels['input_ids'])

    def get_batch_data(self, idx):
        """ Batch data """
        return {key: torch.tensor(val[idx]) for key, val in self.texts.items()}

    def get_batch_labels(self, idx):
        """ Batch label """
        return {key: torch.tensor(val[idx]) for key, val in self.labels.items()}

    def __getitem__(self, idx):
        return self.get_batch_data(idx), self.get_batch_labels(idx)


def train_loop(model, df_train, df_val):
    """ Training step """
    train_dataloader = DataLoader(DataSequence(df_train), batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(DataSequence(df_val), batch_size=BATCH_SIZE)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    if use_cuda:
        model = model.cuda()

    best_metric = 0

    for epoch_num in range(EPOCHS):

        model.train()
        total_loss_train = 0

        for train_data, train_label in tqdm(train_dataloader):
            train_label = train_label['input_ids'].to(device)
            mask = train_data['attention_mask'].to(device)
            input_id = train_data['input_ids'].to(device)

            optimizer.zero_grad()
            loss = model(input_id, mask, labels=train_label).loss
            total_loss_train += loss.item()

            loss.backward()  # Update the weights
            optimizer.step()  # Notify optimizer that a batch is done.
            optimizer.zero_grad()  # Reset the optimer

        model.eval()

        total_loss_val = 0
        pred = []
        gt_ = []

        for val_data, val_label in val_dataloader:
            val_label = val_label['input_ids'].to(device)
            mask = val_data['attention_mask'].to(device)
            input_id = val_data['input_ids'].to(device)

            loss = model(input_id, mask, labels=val_label).loss
            total_loss_val += loss.item()

            outputs = model.generate(input_id)
            outputs = tokenizer.batch_decode(outputs, skip_special_tokens=False)

            labels = tokenizer.batch_decode(val_label, skip_special_tokens=False)

            gt_ = gt_ + extract_triple(labels, gold_extraction=True)
            pred = pred + extract_triple(outputs, gold_extraction=False)

            del outputs, labels
        combined_metric = 0

        scores, precision, recall, f1_score = re_score(pred, gt_, 'relation')
        combined_metric += scores["ALL"]["Macro_f1"]

        scores, precision, recall, f1_score = re_score(pred, gt_, 'subject')
        combined_metric += scores["ALL"]["Macro_f1"]

        scores, precision, recall, f1_score = re_score(pred, gt_, 'object')
        combined_metric = (combined_metric + scores["ALL"]["Macro_f1"]) / 3

        best_metric = check_best_performing(model, best_metric, combined_metric, SAVE_PATH)
        del scores, precision, recall, f1_score

        if best_metric == combined_metric:
            STATS_REPORTS['epoch'] = epoch_num + 1
            STATS_REPORTS['loss'] = f'{total_loss_train / len(df_train): .6f}'
            STATS_REPORTS['val_loss'] = f'{total_loss_val / len(df_val): .6f}'

        print(
            f'Epochs: {epoch_num + 1} | ',
            f'Loss: {total_loss_train / len(df_train): .6f} | ',
            f'Val_Loss: {total_loss_val / len(df_val): .6f}')


def extract_triple(texts, gold_extraction, prediction=False):
    """ Extract triple from prediction output """
    triplets = []
    for text in texts:
        try:
            text = ''.join(text).replace('<s>', '').replace('</s>', '').replace('<pad>', '')
            relation = ''
            for token in text.split():
                if token == "<triplet>":
                    current = 't'
                    if relation != '':
                        triplets.append((subject.strip(), relation.strip(), object_.strip()))
                        relation = ''
                    subject = ''
                elif token == "<subj>":
                    current = 's'
                    if relation != '':
                        triplets.append((subject.strip(), relation.strip(), object_.strip()))
                    object_ = ''
                elif token == "<obj>":
                    current = 'o'
                    relation = ''
                else:
                    if current == 't':
                        subject += ' ' + token
                    elif current == 's':
                        object_ += ' ' + token
                    elif current == 'o':
                        relation += ' ' + token
            triplets.append((subject.strip(), relation.strip(), object_.strip()))
        except Exception as _:
            if gold_extraction:
                print("Gold labels should always be extracted correctly. Exiting")
                sys.exit()
            triplets.append(("Invalid", "Invalid", "Invalid"))

    if prediction: #This is to make sure not more than 1 set of triplets are extracted
        return [triplets[0]]

    return triplets


def re_score(predictions, ground_truths, type):
    """Evaluate RE predictions
    Args:
        predictions (list) :  list of list of predicted relations (several relations in each sentence)
        ground_truths (list) :    list of list of ground truth relations
        type (str) :          the kind of evaluation (relation, subject, object) """
    if type == 'relation':
        # vocab = ['cause', 'enable', 'prevent', 'intend']
        vocab = ['links', 'is a', 'stores', 'a necessary step in the reproduction of', 'used for transferring', 'becomes', 'has', 'surrounds', 'arise by', 'possessing', 'requires', 'in', 'being', 'refers to tendency to move up', 'evolved by means of', 'develops from', 'serve as subunits of', 'dissolved in', 'caused by', 'that occurs directly from the solid phase is called', 'prey', 'consisting of', 'was defined as', 'used', 'of', 'is', 'are synthesized in', 'follow', 'allows for', 'binds to', 'is related to', 'form', 'occurs in', 'hunt', 'propagate', 'give rise to', 'takes on characteristics of', 'includes', 'prey on', 'is form of', 'can be used to generate', 'is divided into', 'is referred to as', 'can be attributed to two or more', 'defines as change over', 'developing out of', 'is in contrast to', 'plays role in', 'increase', 'is movement of solvent across', 'helps explain', 'produced by', 'consist of', 'observed', 'promotes', 'can be produced from', 'enabling', 'eat', 'lie underneath', 'live in', 'preceding', 'include', 'like to eat', 'are sites of', 'are composed of', 'part of', 'is reproductive structure found in', 'can regulate', 'keep in the light', 'prefers', 'approximately equal to half of', 'find applications in', 'keep', 'following', 'have food stores in the form of', 'are prey of', 'attack', 'bonded to', 'is situated in', 'diet consists of', 'predator', 'excludes', 'is substance that', 'having', 'forage', 'can be', 'high', 'are produced by', 'consists of', 'has potential to alter', 'do not have', 'eating', 'consume', 'moves continually through', 'connects', 'starts with', 'are not considered', 'known as', 'components of', 'is phase of', 'can form', 'can be used to predict', 'have one copy of each', 'are made from', 'is commonly referred to', 'is responsible for', 'is tip of', 'serves as site of', 'seeds contain only one', 'live on diet of', 'is composed of', 'example', 'use to provide energy', 'is followed by', 'is paired with', 'dist consists of', 'select', 'receives', 'make up diet', 'are', 'are derived from', 'contain']
        predictions = [pred[1] for pred in predictions]
        ground_truths = [gt_[1] for gt_ in ground_truths]

    elif type == 'subject':
        predictions = [pred[0] for pred in predictions]
        ground_truths = [gt_[0] for gt_ in ground_truths]
        vocab = np.unique(ground_truths).tolist()

    elif type == 'object':
        predictions = [pred[2] for pred in predictions]
        ground_truths = [gt_[2] for gt_ in ground_truths]
        vocab = np.unique(ground_truths).tolist()

    scores = {rel: {"tp": 0, "fp": 0, "fn": 0} for rel in vocab + ["ALL"]}

    # Count GT relations and Predicted relations
    n_sents = len(ground_truths)
    n_rels = n_sents  # Since every 'sentence' has only 1 relation
    n_found = n_sents

    # Count TP, FP and FN per type
    for pred_sent, gt_sent in zip(predictions, ground_truths):
        for entity in vocab:

            if pred_sent == entity:
                pred_entities = {pred_sent}
            else:
                pred_entities = set()

            if gt_sent == entity:
                gt_entities = {gt_sent}

            else:
                gt_entities = set()

            scores[entity]["tp"] += len(pred_entities & gt_entities)
            scores[entity]["fp"] += len(pred_entities - gt_entities)
            scores[entity]["fn"] += len(gt_entities - pred_entities)

    # Compute per relation Precision / Recall / F1
    for entity in scores.keys():
        if scores[entity]["tp"]:
            scores[entity]["p"] = \
                100 * scores[entity]["tp"] / (scores[entity]["fp"] + scores[entity]["tp"])
            scores[entity]["r"] = \
                100 * scores[entity]["tp"] / (scores[entity]["fn"] + scores[entity]["tp"])
        else:
            scores[entity]["p"], scores[entity]["r"] = 0, 0

        if not scores[entity]["p"] + scores[entity]["r"] == 0:
            scores[entity]["f1"] = 2 * scores[entity]["p"] * scores[entity]["r"] / (
                    scores[entity]["p"] + scores[entity]["r"])
        else:
            scores[entity]["f1"] = 0

    # Compute micro F1 Scores
    tp = sum(scores[entity]["tp"] for entity in vocab)
    fp = sum(scores[entity]["fp"] for entity in vocab)
    fn = sum(scores[entity]["fn"] for entity in vocab)

    if tp:
        precision = 100 * tp / (tp + fp)
        recall = 100 * tp / (tp + fn)
        f1_score = 2 * precision * recall / (precision + recall)

    else:
        precision, recall, f1_score = 0, 0, 0

    scores["ALL"]["p"] = precision
    scores["ALL"]["r"] = recall
    scores["ALL"]["f1"] = f1_score
    scores["ALL"]["tp"] = tp
    scores["ALL"]["fp"] = fp
    scores["ALL"]["fn"] = fn

    # Compute Macro F1 Scores
    scores["ALL"]["Macro_f1"] = np.mean([scores[ent_type]["f1"] for ent_type in vocab])
    scores["ALL"]["Macro_p"] = np.mean([scores[ent_type]["p"] for ent_type in vocab])
    scores["ALL"]["Macro_r"] = np.mean([scores[ent_type]["r"] for ent_type in vocab])

    # print(f"RE Evaluation in *** {mode.upper()} *** mode")

    if type == 'relation':
        print(
            "processed {} sentences with {} entities; found: {} relations; correct: {}.".format(n_sents, n_rels,
                                                                                                n_found,
                                                                                                tp))
        # print(
        #     "\tALL\t TP: {};\tFP: {};\tFN: {}".format(
        #         scores["ALL"]["tp"],
        #         scores["ALL"]["fp"],
        #         scores["ALL"]["fn"]))
        # print(
        #     "\t\t(m avg): precision: {:.2f};\trecall: {:.2f};\tf1: {:.2f} (micro)".format(
        #         precision,
        #         recall,
        #         f1))
        # print(
        #     "\t\t(M avg): precision: {:.2f};\trecall: {:.2f};\tf1: {:.2f} (Macro)\n".format(
        #         scores["ALL"]["Macro_p"],
        #         scores["ALL"]["Macro_r"],
        #         scores["ALL"]["Macro_f1"]))

        # for entity in vocab:
        #     print("\t{}: \tTP: {};\tFP: {};\tFN: {};\tprecision: {:.2f};\trecall: {:.2f};\tf1: {:.2f};\t{}".format(
        #         entity,
        #         scores[entity]["tp"],
        #         scores[entity]["fp"],
        #         scores[entity]["fn"],
        #         scores[entity]["p"],
        #         scores[entity]["r"],
        #         scores[entity]["f1"],
        #         scores[entity]["tp"] +
        #         scores[entity][
        #             "fp"]))

    else:
        print(f"Macro F1 for {type}: {scores['ALL']['Macro_f1']:.4f}")
        print(f"Micro F1 for {type}: {scores['ALL']['f1']:.4f}")

    return scores, precision, recall, f1_score


def calc_acc(predictions, gold):
    """ Accuracy """
    num_ner = len(predictions)  # The total number of entities
    acc_subj_correct = 0
    acc_obj_correct = 0

    for pred, gt_ in zip(predictions, gold):
        if pred[0] == gt_[0]:  # The subjects match
            acc_subj_correct += 1

        if pred[2] == gt_[2]:  # The objects match
            acc_obj_correct += 1

    acc_subj_correct = acc_subj_correct / num_ner
    acc_obj_correct = acc_obj_correct / num_ner

    print(f"acc subject: {acc_subj_correct} acc object: {acc_obj_correct}")

    return acc_subj_correct, acc_obj_correct


def check_best_performing(model, best_metric, new_metric, PATH):
    """ Saving model if new best metric """
    if new_metric > best_metric:
        torch.save(model, PATH)
        print("New best model found, saving...")
        best_metric = new_metric
        STATS_REPORTS["best_metric"] = best_metric
    return best_metric


def test_model(data, model):
    """ Predict on test set """
    test_dataset = DataSequence(data)
    test_dataloader = DataLoader(test_dataset, batch_size=4)
    model.eval()

    pred = []
    gt_ = []

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        model = model.cuda()

    for val_data, val_label in test_dataloader:
        test_label = val_label['input_ids'].to(device)
        input_id = val_data['input_ids'].to(device)

        outputs = model.generate(input_id)
        outputs = tokenizer.batch_decode(outputs, skip_special_tokens=False)
        labels = tokenizer.batch_decode(test_label, skip_special_tokens=False)

        gt_ = gt_ + extract_triple(labels, gold_extraction=True)
        pred = pred + extract_triple(outputs, gold_extraction=False)

        del outputs, labels

    scores, precision, recall, f1_score = re_score(pred, gt_, 'relation')
    STATS_REPORTS["test_relation"] = {"scores": scores, "precision": precision,
                                      "recall": recall, "f1_score": f1_score}
    scores, precision, recall, f1_score = re_score(pred, gt_, 'subject')
    STATS_REPORTS["test_subject"] = {"scores": scores, "precision": precision,
                                      "recall": recall, "f1_score": f1_score}
    scores, precision, recall, f1_score = re_score(pred, gt_, 'object')
    STATS_REPORTS["test_object"] = {"scores": scores, "precision": precision,
                                      "recall": recall, "f1_score": f1_score}



def make_predictions(texts, path_to_model):
    """

    :param texts: List of sentences
    :param path_to_model: The path to the model
    :return: List of original sentences and their predictions
    """

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model = torch.load(path_to_model).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained('Babelscape/rebel-large')

    results = []
    for sentence in texts:

        encoding = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True).to(device)
        outputs = model.generate(**encoding, do_sample=True)
        outputs = tokenizer.batch_decode(outputs, skip_special_tokens=False)

        results+= (extract_triple(outputs, gold_extraction=False, prediction=True))

    return texts, results

if __name__ == "__main__":
    #data = pd.read_csv('drive/MyDrive/rebel_format_v2.csv')
    # df_train, df_val = train_test_split(data, test_size=0.1, random_state=SEED)
    DF_TRAIN = pd.read_csv('src/fine_tune_rebel/cm_biology_train.csv')
    DF_VAL = pd.read_csv('src/fine_tune_rebel/cm_biology_eval.csv')
    #del data

    MODEL_CHECKPOINT = "Babelscape/rebel-large"
    MODEL = AutoModelForSeq2SeqLM.from_pretrained(MODEL_CHECKPOINT)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

    train_loop(MODEL, DF_TRAIN, DF_VAL)
    f = open("src/fine_tune_rebel/finetuning_report.txt", "w+", encoding='utf-8')
    f.write("TRAINING & EVAL\n\n")
    f.write(f"Best metric: {STATS_REPORTS['best_metric']}\n")
    f.write(f"Epoch: {STATS_REPORTS['epoch']}\n")
    f.write(f"Training loss: {STATS_REPORTS['loss']}\n")
    f.write(f"Validation loss: {STATS_REPORTS['val_loss']}\n======\n\n======\n\n")

    test_data = pd.read_csv('src/fine_tune_rebel/cm_biology_test.csv')
    MODEL = torch.load('src/fine_tune_rebel/finetuned_rebel.pth')
    test_model(test_data, MODEL)

    f.write("TEST\n\n")
    for x in ["relation", "subject", "object"]:
        key = f'test_{x}'
        f.write(f"{x}\n-----\n")
        f.write(f"Scores: {STATS_REPORTS[key]['scores']}\n")
        f.write(f"Precision: {STATS_REPORTS[key]['precision']}\n")
        f.write(f"Recall: {STATS_REPORTS[key]['recall']}\n")
        f.write(f"F1: {STATS_REPORTS[key]['f1_score']}\n")
        f.write("======\n\n")
    f.close()

    texts, results = make_predictions(texts=test_data.context.to_list(), path_to_model='src/fine_tune_rebel/finetuned_rebel.pth')
    pd.DataFrame({"context": texts, "triplets": results}).to_csv("src/fine_tune_rebel/cm_biology_test_predicted.csv")
