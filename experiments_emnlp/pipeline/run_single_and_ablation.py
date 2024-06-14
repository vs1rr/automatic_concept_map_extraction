# -*- coding: utf-8 -*-
"""
Run experiments on the Wiki train or test dataset
--> fixed parameters

RE = relation extraction

Also running ablation studies
- summary + RE
- importance + RE
- entity + RE
- RE
"""
import os
import click
from src.experiment import ExperimentRun
from src.settings import API_KEY_GPT

####### PARAMS BELOW TO UPDATE 
TYPE_DATA = "multi"
ONE_CM = False
####################

FIXED_PARAMS = {
    # PREPROCESSING
    "preprocess": True,
    "spacy_model": "en_core_web_lg",
    # SUMMARISATION
    "summary_percentage": 15,
    "summary_method": "chat-gpt",
    # IMPORTANCE RANKING
    "ranking_how": "all",
    "ranking_perc_threshold": 0.15,
    # ENTITY
    "confidence": 0.7,
    "options_ent": ["dbpedia_spotlight"],
    "db_spotlight_api": "http://localhost:2222/rest/annotate",
    # RELATION
    "rebel_tokenizer": "Babelscape/rebel-large",
    "summary_how": "single",
    "api_key_gpt": API_KEY_GPT,
    "engine": "gpt-3.5-turbo",
    "temperature": 0.0,
    "options_rel": ["rebel"],
    "local_rm": False,
    "rebel_model": "Babelscape/rebel-large",
    # POSTPROCESS
    "postprocess": True,
}

VARIABLE_PARAMS = {
    "ranking": {"all": "word2vec", "ir": "page_rank"}
}

def init_exps(data_path, summary_folder):
    """
    Returning a list of exps:
    (all, summary+entity+RE, importance+entity+RE)
    """
    res = []
    experiment = ExperimentRun(
        # DATA INFO
        folder_path=data_path, type_data=TYPE_DATA, one_cm=ONE_CM,
        summary_path=os.path.join(
            summary_folder, FIXED_PARAMS["summary_method"], str(FIXED_PARAMS["summary_percentage"])),
        # PREPROCESS
        preprocess=FIXED_PARAMS["preprocess"],
        spacy_model=FIXED_PARAMS["spacy_model"],
        # SUMMARY
        summary_how=FIXED_PARAMS["summary_how"],
        summary_method=FIXED_PARAMS["summary_method"],
        api_key_gpt=API_KEY_GPT,
        engine=FIXED_PARAMS["engine"],
        temperature=FIXED_PARAMS["temperature"],
        summary_percentage=FIXED_PARAMS["summary_percentage"],
        # IMPORTANCE RANKING
        ranking=VARIABLE_PARAMS["ranking"]["all"],
        ranking_how=FIXED_PARAMS["ranking_how"],
        ranking_perc_threshold=FIXED_PARAMS["ranking_perc_threshold"],
        # ENTITY
        options_ent=FIXED_PARAMS["options_ent"],
        confidence=FIXED_PARAMS["confidence"],
        db_spotlight_api=FIXED_PARAMS["db_spotlight_api"],
        # RELATION EXTRACTION
        options_rel=FIXED_PARAMS["options_rel"],
        rebel_tokenizer=FIXED_PARAMS["rebel_tokenizer"],
        rebel_model=FIXED_PARAMS["rebel_model"],
        local_rm=FIXED_PARAMS["local_rm"],
        # POSTPROCESS
        postprocess=FIXED_PARAMS["postprocess"])
    res.append(experiment)
    experiment = ExperimentRun(
        # DATA INFO
        folder_path=data_path, type_data=TYPE_DATA, one_cm=ONE_CM,
        summary_path=os.path.join(
            summary_folder, FIXED_PARAMS["summary_method"], str(FIXED_PARAMS["summary_percentage"])),
        # PREPROCESS
        preprocess=FIXED_PARAMS["preprocess"],
        spacy_model=FIXED_PARAMS["spacy_model"],
        # SUMMARY
        summary_how=FIXED_PARAMS["summary_how"],
        summary_method=FIXED_PARAMS["summary_method"],
        api_key_gpt=API_KEY_GPT,
        engine=FIXED_PARAMS["engine"],
        temperature=FIXED_PARAMS["temperature"],
        summary_percentage=FIXED_PARAMS["summary_percentage"],
        # ENTITY
        options_ent=FIXED_PARAMS["options_ent"],
        confidence=FIXED_PARAMS["confidence"],
        db_spotlight_api=FIXED_PARAMS["db_spotlight_api"],
        # RELATION EXTRACTION
        options_rel=FIXED_PARAMS["options_rel"],
        rebel_tokenizer=FIXED_PARAMS["rebel_tokenizer"],
        rebel_model=FIXED_PARAMS["rebel_model"],
        local_rm=FIXED_PARAMS["local_rm"],
        # POSTPROCESS
        postprocess=FIXED_PARAMS["postprocess"])
    res.append(experiment)
    experiment = ExperimentRun(
        # DATA INFO
        folder_path=data_path, type_data=TYPE_DATA, one_cm=ONE_CM,
        # PREPROCESS
        preprocess=FIXED_PARAMS["preprocess"],
        spacy_model=FIXED_PARAMS["spacy_model"],
        # IMPORTANCE RANKING
        ranking=VARIABLE_PARAMS["ranking"]["ir"],
        ranking_how=FIXED_PARAMS["ranking_how"],
        ranking_perc_threshold=FIXED_PARAMS["ranking_perc_threshold"],
        # ENTITY
        options_ent=FIXED_PARAMS["options_ent"],
        confidence=FIXED_PARAMS["confidence"],
        db_spotlight_api=FIXED_PARAMS["db_spotlight_api"],
        # RELATION EXTRACTION
        options_rel=FIXED_PARAMS["options_rel"],
        rebel_tokenizer=FIXED_PARAMS["rebel_tokenizer"],
        rebel_model=FIXED_PARAMS["rebel_model"],
        local_rm=FIXED_PARAMS["local_rm"],
        # POSTPROCESS
        postprocess=FIXED_PARAMS["postprocess"])
    res.append(experiment)
    return res


@click.command()
@click.argument("data_path")
@click.argument("save_folder")
@click.argument("summary_folder")
def main(data_path, save_folder, summary_folder):
    exps = init_exps(data_path, summary_folder)
    run_exps = [x for x in os.listdir(save_folder) if os.path.isdir(os.path.join(save_folder, x))]
    run_exps = [x.split("-")[-1] for x in run_exps]
    for exp, name in zip(exps, ["all", "summary+entity+re", "ir+entity+re"]):
        if name not in run_exps:
            exp(save_folder=save_folder)

            exp_run = sorted(os.listdir(save_folder))[-1]
            os.rename(os.path.join(save_folder, exp_run), os.path.join(save_folder, f"{exp_run}-{name}"))



if __name__ == '__main__':
    main()
