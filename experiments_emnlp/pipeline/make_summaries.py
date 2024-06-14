# -*- coding: utf-8 -*-
"""
To avoid running GPT over and over, pre-storing them for the experiments
"""
import spacy
import click
import os
import json
from tqdm import tqdm
from loguru import logger
from datetime import datetime
from src.data_load import DataLoader
from src.preprocess import PreProcessor
from src.summary import TextSummarizer
from src.settings import API_KEY_GPT

def update_dir(input_dir):
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)

def make_dirs(root_folder, summary_options, perc_options):
    update_dir(root_folder)
    for s_o in summary_options:
        update_dir(os.path.join(root_folder, s_o))
        for p_o in perc_options:
            update_dir(os.path.join(root_folder, s_o, str(p_o)))

def read_file(path):
    with open(path, "r", encoding="utf-8") as file:
        content = file.read()
    return content

def read_json(json_path):
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as openfile:
            data = json.load(openfile)
        return data
    return {}

def save_json(json_path, data):
    with open(json_path, "w", encoding="utf-8") as openfile:
        json.dump(data, openfile, indent=4)

@click.command()
@click.argument("data_path")
@click.argument('root_folder')
@click.argument('type_data')
@click.argument('dataset')
def main(data_path, root_folder, type_data, dataset):
    types_data = ["train", "test"]
    if type_data not in types_data:
        raise ValueError(f"`type_data` must be in {types_data}")
    datasets = ["wiki", "bio"]
    if dataset not in datasets:
        raise ValueError(f"`dataset` must be in {datasets}")
    if dataset == "wiki":
        if type_data == "train":
            summary_options = ["chat-gpt", "lex-rank"]
            perc_options = [5, 15, 30, 50, 70]
        else:  # type_data == "test"
            summary_options = ["chat-gpt"]
            perc_options = [15]
        data_loader = DataLoader(path=data_path, type_d="multi", one_cm=False)
    else:
        summary_options = ["chat-gpt"]
        perc_options = [15]
        data_loader = DataLoader(path=data_path, type_d="single", one_cm=False)

    make_dirs(root_folder, summary_options, perc_options)

    spacy_model = "en_core_web_lg"
    
    preprocessor = PreProcessor(model=spacy_model)
    nlp = spacy.load(spacy_model)
    nb_folder = len(data_loader.files)
    json_path = os.path.join(root_folder, "logs.json")
    logs = read_json(json_path)

    # Retrieving summaries for each (1) summary option (2) percentage option (3) folder (4) file
    for s_o in summary_options:
        logs[s_o] = {} if s_o not in logs else logs[s_o]
        for p_o in perc_options:
            logs[s_o][p_o] = {} if p_o not in logs[s_o] else logs[s_o][p_o]
            logger.info(f"Summaries\tMethod: {s_o} | Perc: {str(p_o)}")
            summariser = TextSummarizer(method=s_o, api_key_gpt=API_KEY_GPT,
                                        engine="gpt-3.5-turbo-0125", temperature=0.0,
                                        summary_percentage=p_o)

            for i_folder, folder_info in enumerate(data_loader.files):
                folder = folder_info['folder']
                logs[s_o][p_o][folder] = {} if folder not in logs[s_o][p_o] else logs[s_o][p_o][folder]
                logger.info(f"Folder {i_folder+1} ({round(100*(i_folder+1)/nb_folder)}%)")
                update_dir(os.path.join(root_folder,s_o, str(p_o), folder))

                for name, path in tqdm(folder_info["text"]):
                    save_path = os.path.join(root_folder, s_o, str(p_o), folder, f"{name}.txt")
                    if not os.path.exists(save_path):
                        start_time = datetime.now()
                        content = read_file(path)
                        doc = nlp(content)
                        input_data = [sent.text.strip() for sent in doc.sents]
                        input_data = [preprocessor(x) for x in input_data]
                        text = "\n".join(input_data)
                        summary = summariser(text)
                        doc = nlp(summary)
                        end_time = datetime.now()
                        logs[s_o][p_o][folder][name] = {
                            "start": str(start_time), "end": str(end_time),
                            "total": str(end_time-start_time)}

                        f = open(save_path, "w+")
                        for sent in doc.sents:
                            f.write(f"{sent.text}\n")
                        f.close()

                        save_json(json_path=json_path, data=logs)

if __name__ == '__main__':
    main()