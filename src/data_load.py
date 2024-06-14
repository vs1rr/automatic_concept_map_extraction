"""
Helpers for loading data
"""
import os
from typing import Union

class DataLoader:
    """ Handling data load """
    def __init__(self, path: str, type_d: str, one_cm: bool,
                 summary_path: Union[str, None] = None):
        """
        - path: path to folder
        - type_d: either single or multi document 
        - one_cm: one cm or several
        """
        type_d_p = ["single", "multi"]
        if type_d not in type_d_p:
            raise ValueError(f"`type_d` must be in {type_d_p}")
        self.path = path

        if not isinstance(one_cm, bool):
            raise ValueError("`one_cm` must be boolean")

        self.params = {"path": path, "type_d": type_d,
                       "one_cm": one_cm, "summary_path": summary_path}

        # Handling multi-document
        if type_d == "multi":
            if one_cm:
                self.files = [self.get_md_one_folder(path=path, folder=path.split("/")[-1])]
            else:
                directories = [(os.path.join(path, x), x) for x in os.listdir(path) \
                    if os.path.isdir(os.path.join(path, x))]
                self.files = [self.get_md_one_folder(path=path, folder=x) for path, x in directories]

        # Handling single-document
        if type_d == "single":
            if one_cm:
                self.files = [self.get_sd_one_folder(path=path, folder=path.split("/")[-1])]
            else:
                directories = [(os.path.join(path, x), x) for x in os.listdir(path) \
                    if os.path.isdir(os.path.join(path, x))]
                self.files = [self.get_sd_one_folder(path=path, folder=x) for path, x in directories]

        self.summaries = self.get_summaries(summary_path=summary_path, one_cm=one_cm) if summary_path else None

    def get_summaries(self, summary_path: str, one_cm: bool):
        if one_cm:
            folder = summary_path.split("/")[-1]
            res = {folder: self.get_summaries_one_folder(summary_path)}
            
        else:  # one_cm == False
            res = {}
            info = [(x, os.path.join(summary_path, x)) for x in os.listdir(summary_path)]
            info = [x for x in info if os.path.isdir(x[1])]

            for folder, folder_dir in info:
                res[folder] = self.get_summaries_one_folder(folder_dir)
            
        return res
    
    @staticmethod
    def get_summaries_one_folder(path: str):
        return {
                os.path.splitext(x)[0]: os.path.join(path, x) \
                    for x in os.listdir(path) if x.endswith(".txt")
            }

    @staticmethod
    def get_md_one_folder(path: str, folder: str):
        """ Get files from one folder for multi document CM """
        files = [x for x in os.listdir(path) if x.endswith(".cmap") or x.endswith(".txt")]
        gs = [os.path.join(path, x) for x in files if x.endswith(".cmap")][0]
        texts = [(os.path.splitext(x)[0], os.path.join(path, x)) \
            for x in files if x.endswith(".txt")]
        return {"gs": gs, "text": texts, "folder": folder}

    @staticmethod
    def get_sd_one_folder(path: str, folder: str):
        """ Get files from one folder for single document CM """
        curr_folder = os.path.join(path, "gold")
        gs = [os.path.join(curr_folder, x) for x in os.listdir(curr_folder) if x.endswith(".tsv")][0]
        curr_folder = os.path.join(path, "text")
        text = [(os.path.splitext(x)[0], os.path.join(curr_folder, x)) for x in os.listdir(curr_folder) if x.endswith(".txt")]
        return {"gs": gs, "text": text, "folder": folder}


if __name__ == '__main__':
    DATA_LOADER = DataLoader(
        path="./src/data/Corpora_Falke/Wiki/train/101",
        type_d="multi", one_cm=True)
    print(f"MULTI - ONE CM\n{DATA_LOADER.files}\n==========")

    DATA_LOADER = DataLoader(
        path="./src/data/Corpora_Falke/Wiki/train",
        type_d="multi", one_cm=False)
    print(f"MULTI - MULTI CM\n{DATA_LOADER.files}\n==========")

    DATA_LOADER = DataLoader(
        path="./src/data/Corpora_Falke/Biology/topic_1",
        type_d="single", one_cm=True)
    print(f"SINGLE - ONE CM\n{DATA_LOADER.files}\n==========")

    DATA_LOADER = DataLoader(
        path="./src/data/Corpora_Falke/Biology",
        type_d="single", one_cm=False)
    print(f"SINGLE - MULTI CM\n{DATA_LOADER.files}\n==========")


# from src.data_load import DataLoader
# path="./src/data/Corpora_Falke/Wiki/train/101"
# type_d="multi"
# one_cm=False
# path="./src/data/Corpora_Falke/Wiki/train"
# summary_path = "./summaries/chat-gpt/50"
# dl=DataLoader(path=path, type_d=type_d, one_cm=one_cm, summary_path=summary_path)