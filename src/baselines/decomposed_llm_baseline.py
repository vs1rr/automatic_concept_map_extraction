# -*- coding: utf-8 -*-
"""
LLM Baselines for CM building
"""
import os
import json
import spacy
from typing import List, Union
from tqdm import tqdm
from loguru import logger
from openai import OpenAI
import pandas as pd
from src.data_load import DataLoader
from src.preprocess import PreProcessor
from src.evaluation import EvaluationMetrics
from src.settings import API_KEY_GPT

CLIENT = OpenAI(api_key=API_KEY_GPT)
MODEL = "gpt-3.5-turbo-0125"

def get_texts_from_folder(folder):
    """ Retrieves all .txt file in folder """
    res = [x for x in os.listdir(folder) if x.endswith(".txt")]
    return res, [open(os.path.join(folder, x), encoding="utf-8").read() for x in res]

def run_gpt(prompt: str, content: Union[str, List[str]], **add_content):
    """ Get answer from GPT from prompt + content """
    if isinstance(content, str):
        content = " ".join(content.split()[:10000])
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": content}
        ]
    else:  # list of text
        len_prompt = len(prompt.split())
        messages = [{"role": "system", "content": prompt}]
        for c in content:
            if 10000 - len_prompt > len(c.split()):
                messages.append({"role": "user", "content": c})
                len_prompt += len(c.split())
            else:
                break

    if add_content and add_content.get("entities"):
        messages += [{"role": "user", "content": 'The entities are:\n' + \
            add_content.get("entities")}]
    completion = CLIENT.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0)
    return completion.choices[0].message.content

def save_to_folder(folder: str, content: Union[List[str], str], names: str):
    """ Save content to folder """
    if not os.path.exists(folder):
        os.makedirs(folder)
    if isinstance(content, list):
        for i, name in enumerate(names):
            f = open(os.path.join(folder, name), 'w', encoding='utf-8')

            if "```csv" in content[i]:
                for line in content[i].replace("```csv", "").replace("```", "").strip().split("\n"):
                    if len(line.split(';')) == 3:
                        f.write(line + "\n")
            else:
                f.write(content[i].strip())
            f.close()
    else:
        f = open(os.path.join(folder, names[0]), 'w', encoding='utf-8')
        if "```csv" in content:
            for line in content.replace("```csv", "").replace("```", "").strip().split("\n"):
                if len(line.split(';')) == 3:
                    f.write(line + "\n")
        else:
            f.write(content.strip())
        f.close()

def get_gs_triples(file_path):
    """ Get ground truth triples """
    res = open(file_path, "r", encoding='utf-8').readlines()
    return [x.replace("\n", "").split("\t") for x in res]

class DecomposedLLMBaseline:
    """ LLM Baseline (using one model of OpenAI) for CM extraction """
    def __init__(self, model: str = MODEL):
        self.model = model
        self.preprocess = PreProcessor()

        self.prompt_summary = """
        You need to write a summary of the text that will be sent in the following message. The summary should be in the same style as the original text. For instance, focus on the content of the text, and do not start with "this text is about" or equivalent.

        The summary is:
        """
        self.prompt_entity = """
        You need to extract all the entities from the text that will be sent in the following message. You must not introduce a hierarchy over the entities, such as Person or Concept. Each entity should appear in the text as is.

        In your answer, you must give the output in a .csv file with the columns `entity` and `surface`. `entity` contains one label, whereas `surface` contains all the surface forms of that entity. The columns are separated by `;`, and the surface forms in the `surface` column by `,`.

        The output is:
        ```csv
        ```
        """
        self.prompt_group_entity = """
        You need to group all the entities from the .csv that will be sent in the following messages into one file. 

        In your answer, you must give the output in a .csv file with the columns `entity` and `surface`. `entity` contains one label, whereas `surface` contains all the surface forms of that entity. The columns are separated by `;`, and the surface forms in the `surface` column by `,`.

        The output is:
        ```csv
        ```
        """
        self.prompt_relation = """
        You need to extract all the relations from the text that will be sent in the following message. Each relation is in the form of a triple (subject, predicate, object), where subject and object are entities that you identified in the previous step. `subject` and `object` should be from the list of entities you will be sent. Each relation must be unique, no repetitions.

        In your answer, you must give the output in a .csv file with the columns with the columns `subject`,  `predicate` and `object`. The columns are separated by `;`.

        The output is:
        ```csv
        ```
        """
        self.prompt_group_relation = """
        You need to group all the relations from the .csv that will be sent in the following messages. If there are duplicates, only keep one relation and ensure that each relation is unique. 

        In your answer, you must give the output in a .csv file with the columns with the columns `subject`,  `predicate` and `object`. The columns are separated by `;`.

        The output is:
        ```csv
        ```
        """
        self.prompt_ir = """
        You will be provided with a set of triples, where each triple consists of a subject, predicate, and object. Your task is to remove redundant triples, and to extract a subset of these triples that represent the most important information from the original set.

        In your answer, you must give the output in a .csv file with the columns with the columns `subject`,  `predicate` and `object`. The columns are separated by `;`.

        The output is:
        ```csv
        ```
        """

        self.prompt_ir_1 = """
        You will be provided with a set of triples, where each triple consists of a subject, predicate, and object. You need to make the triples more compact, ie. remove redundant information.

        The answer must be in the same format as in the input, as a .csv file with the columns with the columns `subject`,  `predicate` and `object`. The columns are separated by `;`.

        The output is:
        ```csv
        ```
        """

        self.prompt_ir_2 = """
        You will be provided with a set of triples, where each triple consists of a subject, predicate, and object. You need to retrieve the most important triples, ie. summarise the triples content.

        The answer must be in the same format as in the input, as a .csv file with the columns with the columns `subject`,  `predicate` and `object`. The columns are separated by `;`.

        The output is:
        ```csv
        ```
        """

        self.nlp = spacy.load("en_core_web_lg")
        self.ir_o = ["one-step", "two-step"]


    def __call__(self, folder: str, save_folder: Union[str, None] = None, ir: str = "one-step"):
        if ir not in self.ir_o:
            raise ValueError(f"The `ir` param must be in {self.ir_o}")

        logger.info(f"Generating concept maps from texts in folder {folder}")
        names, texts = get_texts_from_folder(folder=folder)

        logger.info("Preprocessing texts")

        # Pre-processing
        texts = [self.nlp(text) for text in texts]
        texts = [[sent.text.strip() for sent in text.sents if sent.text.strip()] for text in texts]
        texts = [[self.preprocess(x) for x in sentences] for sentences in texts] 
        texts = ["\n".join(x) for x in texts]

        if save_folder and not os.path.exists(save_folder):
            os.makedirs(save_folder)

        if save_folder:
            save_to_folder(folder=os.path.join(save_folder, "preprocess"),
                           content=texts, names=names)


        logger.info("Generating summaries")
        summaries = []
        for text in tqdm(texts):
            summaries.append(run_gpt(
                prompt=self.prompt_summary, content=text))
        if save_folder:
            save_to_folder(folder=os.path.join(save_folder, "summary"),
                           content=summaries, names=names)

        logger.info("Extracting entities")
        entities = []
        for text in tqdm(summaries):
            entities.append(run_gpt(
                prompt=self.prompt_entity, content=text))
        if save_folder:
            save_to_folder(folder=os.path.join(save_folder, "entity"),
                           content=entities, names=names)

        logger.info("Grouping entities")
        grouped_entities = run_gpt(
            prompt=self.prompt_group_entity, content=entities)
        if save_folder:
            save_to_folder(folder=os.path.join(save_folder, "grouped_entity"),
                           content=grouped_entities, names=["grouped_entities.csv"])

        logger.info("Extracting relations")
        relations = []
        info = {"entities": grouped_entities}
        for text in tqdm(summaries):
            relations.append(run_gpt(
                prompt=self.prompt_relation, content=text, **info))
        if save_folder:
            save_to_folder(folder=os.path.join(save_folder, "relation"),
                           content=relations, names=names)

        logger.info("Grouping relations")
        grouped_relations = run_gpt(
            prompt=self.prompt_group_relation, content=relations)
        if save_folder:
            save_to_folder(folder=os.path.join(save_folder, "grouped_relation"),
                           content=grouped_relations, names=["grouped_relations.csv"])

        if ir == "one-step":
            logger.info("Extracting most important relations in one step")
            output = run_gpt(
                prompt=self.prompt_ir, content=grouped_relations)
            if save_folder:
                save_to_folder(folder=save_folder,
                               content=output, names=["output.csv"])

        if ir == "two-step":
            logger.info("Making triples more compact")
            compact_triples = run_gpt(
                prompt=self.prompt_ir_1, content=grouped_relations)
            if save_folder:
                save_to_folder(folder=os.path.join(save_folder, "compact_triples"),
                               content=compact_triples, names=["compact_triples.csv"])

            logger.info("Retrieving most important triples")
            output = run_gpt(
                prompt=self.prompt_ir_2, content=compact_triples)
            if save_folder:
                save_to_folder(folder=save_folder,
                               content=output, names=["output.csv"])

        logger.success("Finished process")


class DecomposedLLMExperimentRun:
    """ Running experiment on a dataset """
    def __init__(self, data_path: str, type_d: str = "multi", one_cm: bool = False):
        """ Init dataset """
        self.dataset = DataLoader(
            path=data_path, type_d=type_d, one_cm=one_cm
        )
        self.model = DecomposedLLMBaseline()
        self.evaluation_metrics = EvaluationMetrics()

    def run_evaluation(self, gs_path, rel_path):
        """ Running evaluation metrics """
        gs_triples = get_gs_triples(file_path=gs_path)
        system_triples = pd.read_csv(rel_path, sep=";")
        system_triples = [list(x) for x in system_triples.values]
        system_triples = [x for x in system_triples if all(isinstance(y, str) for y in x)]

        return self.evaluation_metrics(
            triples=system_triples, gold_triples=gs_triples)

    def __call__(self, save_folder, ir = "one-step"):
        """ Running on all files of the dataset """
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        metrics_path = os.path.join(save_folder, "metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, "r", encoding="utf-8") as openfile:
                metrics = json.load(openfile)
        else:
            metrics = {}
        for info in self.dataset.files:
            logger.info(f"Processing folder {info['folder']}")
            folder = os.path.join(self.dataset.path, info["folder"])
            curr_sf = os.path.join(save_folder, info["folder"])
            if not os.path.exists(curr_sf):
                os.makedirs(curr_sf)
                self.model(folder=folder, save_folder=curr_sf, ir=ir)
                curr_metrics = self.run_evaluation(
                    gs_path=info['gs'], rel_path=os.path.join(curr_sf, 'output.csv'))

                metrics[info["folder"]] = curr_metrics
                with open(metrics_path, "w", encoding="utf-8") as openfile:
                    json.dump(metrics, openfile, indent=4)




if __name__ == '__main__':
    # FOLDER = "./data/Corpora_Falke/Wiki/test/102"
    # SAVE_FOLDER = "test"
    # BASELINE = DecomposedLLMBaseline()
    # BASELINE(folder=FOLDER, save_folder=SAVE_FOLDER)

    # FOLDER = "./data/Corpora_Falke/Wiki/train/"
    # EXP = DecomposedLLMExperimentRun(data_path=FOLDER)
    # EXP(save_folder="experiments_emnlp/baselines/cot_baseline/train")

    # FOLDER = "./data/Corpora_Falke/Wiki/test/"
    # EXP = DecomposedLLMExperimentRun(data_path=FOLDER)
    # EXP(save_folder="experiments_emnlp/baselines/decomposed_baseline/test")

    # FOLDER = "./data/Corpora_Falke/Wiki/train/"
    # EXP = DecomposedLLMExperimentRun(data_path=FOLDER)
    # EXP(save_folder="experiments_emnlp/baselines/decomposed_baseline/two-step/train", ir="two-step")

    FOLDER = "./data/Corpora_Falke/Wiki/test/"
    EXP = DecomposedLLMExperimentRun(data_path=FOLDER)
    EXP(save_folder="experiments_emnlp/baselines/decomposed_baseline/two-step/test", ir="two-step")
