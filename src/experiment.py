# -*- coding: utf-8 -*-
"""
Running experiments
"""
import os
import json
import subprocess
from typing import List, Union
from datetime import datetime
from loguru import logger
from tqdm import tqdm
from src.data_load import DataLoader
from src.evaluation import EvaluationMetrics
from src.pipeline import CMPipeline

def get_save_folder():
    """ Save folder """
    date = str(datetime.now())
    return f"{date[:10]}-{date[11:19]}"


def create_folders(folder_path: str):
    """ Create folders to save intermediate steps """
    os.makedirs(folder_path)
    for name in ["preprocess", "entity", "relation", "postprocess"]:
        os.makedirs(os.path.join(folder_path, name))


def save_data(preprocess, entities, relations, postprocess, save_folder, name):
    """ Save intermediate steps data """
    with open(os.path.join(
            save_folder, "relation", f"{name}.txt"), "w", encoding="utf-8") as output_file:
        output_file.write("\n".join([", ".join(map(str, rel)) for rel in relations]))

    with open(os.path.join(
            save_folder, "preprocess", f"{name}.txt"), "w", encoding="utf-8") as output_file:
        output_file.write("\n".join(preprocess))

    with open(os.path.join(
            save_folder, "entity", f"{name}.json"), "w", encoding="utf-8") as openfile:
        json.dump({"entities": entities}, openfile, indent=4)

    with open(os.path.join(
            save_folder, "postprocess", f"{name}.txt"), "w", encoding="utf-8") as openfile:
        openfile.write("\n".join(map(str, postprocess)))

def get_gs_triples(file_path):
    res = open(file_path, "r").readlines()
    return [x.replace("\n", "").split("\t") for x in res]


class ExperimentRun:
    """ Running a full experiment """

    def __init__(self,
                 folder_path: str, type_data: str, one_cm: bool,
                 options_rel: List[str],
                 summary_path: Union[str, None] = None,
                 preprocess: bool = False,
                 postprocess: bool = False,
                 spacy_model: Union[str, None] = None,
                 options_ent: Union[List[str], None] = None,
                 confidence: Union[float, None] = None,
                 db_spotlight_api: Union[str, None] = 'https://api.dbpedia-spotlight.org/en/annotate',
                 threshold: Union[int, None] = None,
                 rebel_tokenizer: Union[str, None] = None,
                 rebel_model: Union[str, None] = None,
                 local_rm: Union[bool, None] = None,
                 summary_how: Union[str, None] = None,
                 summary_method: Union[str, None] = None,
                 api_key_gpt: Union[str, None] = None,
                 engine: Union[str, None] = None,
                 temperature: Union[str, None] = None,
                 summary_percentage: Union[str, None] = None,
                 ranking: Union[str, None] = None,
                 ranking_how: Union[str, None] = None,
                 ranking_int_threshold: Union[int, None] = None,
                 ranking_perc_threshold: Union[float, None] = None,
                 word2vec_model_path: Union[str, None] = None):  # Add summary_parameters
        self.data = DataLoader(path=folder_path, type_d=type_data, one_cm=one_cm,
                               summary_path=summary_path)

        logger.info("Data Loader done!")

        self.pipeline = CMPipeline(
            options_rel=options_rel, preprocess=preprocess, spacy_model=spacy_model,
            options_ent=options_ent, confidence=confidence, db_spotlight_api=db_spotlight_api,
            threshold=threshold,
            rebel_tokenizer=rebel_tokenizer, rebel_model=rebel_model, local_rm=local_rm,  
            summary_how=summary_how,
            summary_method=summary_method,
            api_key_gpt=api_key_gpt,
            engine=engine,
            temperature=temperature,
            summary_percentage=summary_percentage,
            ranking=ranking,
            ranking_how=ranking_how,
            postprocess=postprocess,
            ranking_int_threshold=ranking_int_threshold,
            ranking_perc_threshold=ranking_perc_threshold,
            word2vec_model_path=word2vec_model_path)
        self.evaluation_metrics = EvaluationMetrics()

        self.params = self.pipeline.params

        data = self.data.params
        data.update({"files": self.data.files})
        self.params.update({"data": data})  

    def __call__(self, save_folder: str):
        """ A folder will be created in save_folder to store the results of experiments """
        metrics = {}
        logs = {}
        logger.info(f"Running experiments for the following parameters: {self.params}")
        logger.info(f"Running experiments for the following summaries: {self.data.summaries}")

        # Save folder
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        save_folder = os.path.join(save_folder, get_save_folder())
        os.makedirs(save_folder)

        # Save params
        with open(os.path.join(save_folder, "params.json"), "w", encoding="utf-8") as openfile:
            json.dump(self.params, openfile, indent=4)

        # Run pipeline for each folder 
        nb_folder = len(self.data.files)
        for i_folder, folder_info in enumerate(tqdm(self.data.files)):
            folder = folder_info['folder']
            all_relations = []
            logs[folder] = {}

            curr_folder = os.path.join(save_folder, folder)
            folder_t_log = f"[Folder {folder}][{i_folder+1}/{nb_folder} ({round(100*(i_folder+1)/nb_folder)}%)]"
            logger.info(folder_t_log)
            create_folders(folder_path=curr_folder)

            nb_file = len(folder_info["text"])
            # Open all files to be taken into account for that subfolder
            input_content = [open(path, "r", encoding="utf-8").read() for _, path in folder_info["text"]]

            if self.data.summaries:
                file_order = [x for x, _ in folder_info["text"]]
                summaries_list = [self.data.summaries[folder][x] for x in file_order]
                summaries_list = [open(path, "r", encoding="utf-8").read() for path in summaries_list]
            else:
                summaries_list = None

            # Run pipeline
            start_ = datetime.now()
            logs[folder]["start"] =  str(start_)
            c_relations, c_info = self.pipeline(input_content=input_content, summaries_list=summaries_list, verbose=True)

            save_data(relations=c_relations, preprocess=c_info["text"], entities=c_info["entities"], postprocess=c_relations, save_folder=curr_folder, name=folder)
            logger.info("Pipeline & Preprocessing done")

            all_relations = c_relations

            #  Run evaluation
            gs_triples = get_gs_triples(file_path=folder_info["gs"])
            all_relations = list(set(all_relations))
            curr_metrics = self.evaluation_metrics(
                triples=all_relations, gold_triples=gs_triples)
            metrics[folder] = curr_metrics
            logger.info("Evaluation done, saving metrics..")

            # Save metrics and logs
            with open(os.path.join(save_folder, "metrics.json"),
                        "w", encoding="utf-8") as openfile:
                json.dump(metrics, openfile, indent=4)

            end_ = datetime.now()
            # logs[folder][name].update({"end": str(end_), "total": str(end_-start_)})
            logs[folder].update({"end": str(end_), "total": str(end_-start_)})

            with open(os.path.join(save_folder, "logs.json"),
                        "w", encoding="utf-8") as openfile:
                json.dump(logs, openfile, indent=4)

            logger.info(f"Total execution time: {(end_ - start_).total_seconds():.4f}s")

            # Save word2vec model
            if os.path.exists("word2vec.model"):
                subprocess.call(f"mv word2vec.model {os.path.join(save_folder, folder)}", shell=True)
        
        logs["finished"] = "yes"
        with open(os.path.join(save_folder, "logs.json"),
                  "w", encoding="utf-8") as openfile:
            json.dump(logs, openfile, indent=4)

if __name__ == '__main__':
    from src.settings import API_KEY_GPT
    EXPERIMENTR = ExperimentRun(
        # EXPERIMENT PARAMS
        folder_path="./data/Corpora_Falke/Wiki/test",
        type_data="multi", one_cm=False,
        summary_path="./experiments_emnlp/summaries/wiki_test/chat-gpt/15/",

        # PIPELINE PARAMS
        # PREPROCESS
        preprocess=True,
        spacy_model="en_core_web_lg",
        # SUMMARY
        summary_how = "single", summary_method="chat-gpt",
        api_key_gpt=API_KEY_GPT, engine="gpt-3.5-turbo",
        temperature=0.0, summary_percentage=15,
        # RANKING
        ranking="word2vec", ranking_how="all", ranking_perc_threshold=0.15,
        # ENTITY
        # options_ent=["dbpedia_spotlight"],#"nps"
        # confidence=0.5,
        # db_spotlight_api="http://localhost:2222/rest/annotate",
        # threshold=10,
        # RELATION
        options_rel=["rebel"],
        rebel_tokenizer="Babelscape/rebel-large",
        rebel_model="Babelscape/rebel-large",
        local_rm=False,
        # POSTPROCESS
        postprocess=True,
        )
    print(EXPERIMENTR.params)
    EXPERIMENTR(save_folder="ablations")
