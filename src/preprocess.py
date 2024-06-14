"""
Running spacy pipeline for preprocessing : remove stopwords,punctuation, double spaces and citations
"""
import os
import spacy
import regex as re
from tqdm import tqdm
from loguru import logger
from src.settings import nlp

class PreProcessor:
    """ Main class for preprocessing """
    def __init__(self, model: str = "en_core_web_lg"):
        """ Init main params"""
        self.nlp = spacy.load(model)

    def remove_patterns(self, text: str, patterns):
        """ Remove patterns with regex"""
        for pattern in patterns:
            text = re.sub(pattern, '', text)
        return text

    def find_index(self, doc, keywords: str):
        """ Find index of one token"""
        for keyword in keywords:
            index = next((i for i, token in enumerate(doc) if token.text.lower() == keyword.lower()), None)
            if index is not None:
                return index
        return None

    def find_two_token_index(self, doc, two_tokens: str):
        """ Find two tokens one following each other"""
        for i in range(len(doc) - 1):
            if doc[i].text.lower() == two_tokens[0].lower() and doc[i + 1].text.lower() == two_tokens[1].lower():
                return i
        return None

    def create_new_doc(self, doc, end_index: int):
        if end_index is not None:
            return doc[:end_index]
        else:
            return doc

    def __call__(self, text: str):
        """ Preprocessing one string text """
        patterns_to_remove = [
            r'Manuscripts, Medieval--Ireland.*?Trinity College Library, Dublin\.',
            r'\[Accessed[^\]]+\]',
            r'Slide \d+',
            r'http\S+',
            r'ß\?\?\?t\?\?\?Saßß?t\?\?\?st\?\?\?a\?\?;'
        ]

        text_without_patterns = self.remove_patterns(text=text, patterns= patterns_to_remove)
        doc = self.nlp(text_without_patterns)
        tokens_to_filter = [
            "Note:", "Please be patient until they appear.",
            "Illustration for Alfred Noyes' poem \"A Spell for a Fairy\" in Princess Mary's Gift Book by Claude Shepperson.",
            "Architects & Engineers for 9/11 Truth",
            "Reward: Elusive \"History's Business\" Episodewith Larry SilversteinAE911",
            "Follow Scientific American on Twitter", "@SciAm", "@SciamBlogs",
            "Visit Scientific American.com for the latest in science, health and technology news.",
            "© 2011 Scientific American.com. All rights reserved.", "Edit by Thomas Koitzsch",
            "edit", "For more information, visit www.nationaltrust.org.uk/bodiam-castleor call 01580 831324.",
            "Washington-Centerville Public Library,111 West Spring Valley Road,Centerville, OH 45458,www.wclibrary.info",
            "Information on the facsimile from the company that made the facsimile (In Dutch)",
            "By John L. Cisne Perception, Vol.38 (2009)",
            "[Interview]",
            "Gallery",
            "Website:",
            "Go next[edit]",
            "edit",
            "[]",
            "[read less]",
            "Thank you for listening!",
            "Newspaper Article",
            "From Wikipedia, the free encyclopedia",
            "For more information visit:",
            "Image ID:124398484Copyright: Iryna Rasko Available in high-resolution and several sizes to fit the needs of your project."
        ]
        cleaned_tokens = [token.text for token in doc if not token.is_punct or not tokens_to_filter]
        cleaned_text = nlp(' '.join(cleaned_tokens).lower())
        keywords_to_find = ["contributors", "abbreviations", "sources", "refs", "visited", "link", "references",
                            "source",
                            "directions", "sources", "refs", "references", "ReferencesCole", "bibliography",
                            "BibliographyArmitage", "copyright", "notes", "note", "notes1"]

        for keyword in keywords_to_find:
            keyword_index = self.find_index(doc=cleaned_text, keywords =[keyword])

            if keyword_index is not None:
                new_doc = self.create_new_doc(doc=doc, end_index = keyword_index)
                new_text = new_doc.text
                # print(f"{keyword.capitalize()} found. New document:")
                # print(new_text)
            else:
                new_doc = doc
                new_text = new_doc.text
                # print(f"Pre-processing the text.")

        two_token_keywords_to_find = [["more", "information"], ["further", "information"], ["see", "also"],
                                      ["about", "the", "water"], ["go", "to", "text"]]

        for keywords in two_token_keywords_to_find:
            keyword_index = self.find_two_token_index(doc=doc, two_tokens=keywords)

            if keyword_index is not None:
                new_doc = self.create_new_doc(doc=doc, end_index=keyword_index)
                new_text = new_doc.text
                # print(f"{' '.join(keywords).capitalize()} found. New document:")
                # print(new_text)
            else:
                new_doc = doc
                new_text = new_doc.text
                # print(f"Pre-processing the text.")

        cleaned_text_1 = re.sub(r'\s+', ' ', new_text)
        cleaned_text = re.sub(r'\[\d+\]\[\d+\]\[\d+\]', '', cleaned_text_1)

        return cleaned_text

    def get_sentences(self, file_path: str):
        """ Self explanatory """
        with open(file_path, "r", encoding='utf-8') as file:
            text = file.read()
            doc = self.nlp(text)
        return [sent.text.strip() for sent in doc.sents]

    def main_folder(self, input_folder: str, output_folder: str):
        """ Instead of one text, running for one folder"""
        for root, _, files in os.walk(input_folder):
            logger.info(f"[Preprocessing] Root: {root}")
            for file_name in tqdm([x for x in files if x.endswith(".txt")]):
                file_path = os.path.join(root, file_name)
                sentences = self.get_sentences(file_path=file_path)

                cleaned_text = [self(sent) for sent in sentences]
                my_string = '\n'.join(cleaned_text)

                output_root = root.replace(input_folder, output_folder)
                os.makedirs(output_root, exist_ok=True)

                output_file_name = file_name.replace(".txt", "-preprocessed.txt")
                output_file_path = os.path.join(output_root, output_file_name)

                with open(output_file_path, "w", encoding='utf-8') as output_file:
                    output_file.write(my_string)
