# -*- coding: utf-8 -*-
"""
Running additional experiments before the rebuttal phase

Main novelties
- Added another option for relation extraction: `corenlp`
- Added anotion option for entity extraction: `nps` (spacy noun phrases + threshold)

Fixed parameters for these experiments
- Summarisation
    * summary_method -> ChatGPT
- Ranking
    * ranking method -> PageRank
    * ranking how -> all
- Entity
    * If spotlight: confidence -> 0.7
- Relation
    * If rebel: rebel_ft


Additional parameters/settings to test
- rebel_ft + spotlight&nps
- corenlp + spotlight&nps
- rebel_ft&corenlp + spotlight&nps
-> from this get best params

- summarisation only: 15/30/50 + best params
- ranking only: 15/30/50 + best params

Done
- Entity NPs threshold: 1, 2, 3, 5 (also include corenlp+nps)

"""
import os
import click
from src.experiment import ExperimentRun
from src.settings import API_KEY_GPT

# ENTITY_THRESHOLDS = [10, 20, 30, 50]
ENTITY_THRESHOLDS = [1, 2, 3, 5]
ENT_SUMMARY_PERC = [15, 30]
ENT_RANKING_PERC = [0.15, 0.3]

def run_entity_threshold(data_path, save_folder, summary_folder):
    """ Only seeing the impact of the threshold on performance 
    - Fixed parameters as described above
    - Other fixed are:
        * Summarisation + Ranking: 15/30
        * Relation extraction corenlp 
        * Entity: nps """
    for summary_perc in ENT_SUMMARY_PERC:
        for ranking_perc in ENT_RANKING_PERC:
            summary_path = os.path.join(summary_folder, str(summary_perc))
            params = {
                "folder_path": data_path, "type_data": "multi", "one_cm": False, "summary_path": summary_path,
                "preprocess": True, "spacy_model": "en_core_web_lg", "options_ent": ["nps"], "confidence": 0.7,
                "db_spotlight_api": "http://localhost:2222/rest/annotate", "options_rel": ["corenlp"],
                "summary_how": "single", "summary_method": "chat-gpt", "api_key_gpt": API_KEY_GPT,
                "engine": "gpt-3.5-turbo", "temperature": 0.0, "summary_percentage": summary_perc,
                "ranking": "page_rank", "ranking_how": "all", "ranking_perc_threshold": ranking_perc}
            for threshold in ENTITY_THRESHOLDS:
                try:
                    params["threshold"] = threshold
                    experiment = ExperimentRun(**params)
                    print(experiment.pipeline.entity.options)
                    experiment(save_folder=save_folder)
                    exp_run = sorted(os.listdir(save_folder))[-1]
                    os.rename(os.path.join(save_folder, exp_run), os.path.join(save_folder, f"{exp_run}-entity-threshold-{threshold}-summary-{summary_perc}-ranking-{ranking_perc}"))
                except Exception as e:
                    print(e)


def run_entity_all(data_path, save_folder, summary_folder):
    summary_perc, ranking_perc = 15, 0.15
    summary_path = os.path.join(summary_folder, str(summary_perc))
    params = {
        "folder_path": data_path, "type_data": "multi", "one_cm": False, "summary_path": summary_path,
        "preprocess": True, "spacy_model": "en_core_web_lg", "options_ent": ["nps", "dbpedia_spotlight"], "confidence": 0.7,
        "db_spotlight_api": "http://localhost:2222/rest/annotate", 
        "summary_how": "single", "summary_method": "chat-gpt", "api_key_gpt": API_KEY_GPT,
        "engine": "gpt-3.5-turbo", "temperature": 0.0, "summary_percentage": summary_perc,
        "ranking": "page_rank", "ranking_how": "all", "ranking_perc_threshold": ranking_perc}
    options_relations = [["corenlp"], ["rebel"]]
    for index, options_rel in enumerate(options_relations):
        if index == 1:
            curr_params = params
            curr_params["options_rel"] = options_rel
        else:
            curr_params = params
            curr_params.update({
                "options_rel": options_rel,
                "rebel_tokenizer": "Babelscape/rebel-large",
                "rebel_model": "./src/fine_tune_rebel/finetuned_rebel.pth", "local_rm": True
            })
        try:
            experiment = ExperimentRun(**curr_params)
            experiment(save_folder=save_folder)
            exp_run = sorted(os.listdir(save_folder))[-1]
            os.rename(os.path.join(save_folder, exp_run), os.path.join(save_folder, f"{exp_run}-entity-all-{options_rel[0]}"))
        except Exception as e:
            print(e)

def run_entity_relation_all(data_path, save_folder, summary_folder):
    summary_perc, ranking_perc = 15, 0.15
    summary_path = os.path.join(summary_folder, str(summary_perc))
    threshold = 1
    params = {
        "folder_path": data_path, "type_data": "multi", "one_cm": False, "summary_path": summary_path,
        "preprocess": True, "spacy_model": "en_core_web_lg", "options_ent": ["nps", "dbpedia_spotlight"], "threshold": threshold, "confidence": 0.7,
        "db_spotlight_api": "http://localhost:2222/rest/annotate", 
        "summary_how": "single", "summary_method": "chat-gpt", "api_key_gpt": API_KEY_GPT,
        "engine": "gpt-3.5-turbo", "temperature": 0.0, "summary_percentage": summary_perc,
        "ranking": "page_rank", "ranking_how": "all", "ranking_perc_threshold": ranking_perc, "options_rel": ["rebel", "corenlp"],
        "rebel_tokenizer": "Babelscape/rebel-large",
        "rebel_model": "./src/fine_tune_rebel/finetuned_rebel.pth", "local_rm": True}
    try:
        experiment = ExperimentRun(**params)
        experiment(save_folder=save_folder)
        exp_run = sorted(os.listdir(save_folder))[-1]
        os.rename(os.path.join(save_folder, exp_run), os.path.join(save_folder, "{exp_run}-entity-relation-all"))
    except Exception as e:
        print(e)

def run_summarisation_only(data_path, save_folder, summary_folder):
    threshold = 1
    options_ent = ["nps", "dbpedia_spotlight"]
    options_rel = ["rebel", "corenlp"]
    params = {
        "folder_path": data_path, "type_data": "multi", "one_cm": False, 
        "preprocess": True, "spacy_model": "en_core_web_lg", "options_ent": options_ent, "threshold": threshold, "confidence": 0.7,
        "db_spotlight_api": "http://localhost:2222/rest/annotate", 
        "summary_how": "single", "summary_method": "chat-gpt", "api_key_gpt": API_KEY_GPT,
        "engine": "gpt-3.5-turbo", "temperature": 0.0, "options_rel": options_rel,
        "rebel_tokenizer": "Babelscape/rebel-large",
        "rebel_model": "./src/fine_tune_rebel/finetuned_rebel.pth", "local_rm": True}
    for summary_perc in [15, 30, 50]:
        summary_path = os.path.join(summary_folder, str(summary_perc))
        params["summary_path"] = summary_path
        params["summary_percentage"] = summary_perc
        try:
            experiment = ExperimentRun(**params)
            experiment(save_folder=save_folder)
            exp_run = sorted(os.listdir(save_folder))[-1]
            os.rename(os.path.join(save_folder, exp_run), os.path.join(save_folder, f"{exp_run}-summary-only-{summary_perc}"))
        except Exception as e:
            print(e)

def run_ranking_only(data_path, save_folder, summary_folder):
    return


@click.command()
@click.argument("data_path")
@click.argument("save_folder")
@click.argument("summary_folder")
@click.argument("type_exp")
def main(data_path, save_folder, summary_folder, type_exp):
    if type_exp == "entity_threshold":
        run_entity_threshold(data_path, save_folder, summary_folder)


if __name__ == '__main__':
    # python experiments_emnlp/pipeline/run_additional_exps.py data/Corpora_Falke/Wiki/train/ experiments_emnlp/pipeline/additional_exps_wiki_train experiments_emnlp/summaries entity_threshold
    main()