from typing import Union, List
import requests
import spacy
from nltk.corpus import wordnet as wn
from src.settings import nlp
from collections import Counter

class EntityExtractor:
    """ Extracting entities from text """

    def __init__(self, options: List[str] = ["dbpedia_spotlight", "nps","wordnet"],
                 confidence: Union[float, None] = None,
                 db_spotlight_api: str = 'https://api.dbpedia-spotlight.org/en/annotate',
                 threshold: Union[int, None] = 10):
        """ Init main params
        - options: how to extract entities

        Default: calls Spotlight API
        Custom: using local spacy model """
        self.options_to_f = {
            "dbpedia_spotlight": self.get_dbs_ent,
            "wordnet": self.get_wordnet_ent,
            "spacy": self.get_spacy_ent,
            "nps": self.get_np_ent

        }
        self.options_p = list(self.options_to_f.keys())
        self.check_params(options=options, confidence=confidence, threshold=threshold)

        self.params = {
            "options": options,
            "confidence": confidence,
            "db_spotlight_api": db_spotlight_api
        }
        self.options = options

        # DBpedia Spotlight params
        self.confidence = confidence
        self.headers = {'Accept': 'application/json'}
        self.dbpedia_spotlight_api = db_spotlight_api
        self.timeout = 3600

        self.threshold = threshold
        self.nlp = spacy.load("en_core_web_lg")

    def check_params(self, options, confidence, threshold):
        """ Check that each parameter is correct for the options """
        if any(x not in self.options_p for x in options):
            raise ValueError(f"All options in `options` must be from {self.options_p}")

        if "dbpedia_spotlight" in options:
            if not isinstance(confidence, float):
                raise ValueError("To extract entities with DBpedia Spotlight, " + \
                                 "you need to specify `confidence` as a float")
        if threshold and not isinstance(threshold, int):
            raise ValueError("`threshold` param, if not null, must be an integer")
    
    def get_np_ent(self, text: str):
        """ Retrieve entities based on noun phrases + threshold """
        doc = self.nlp(text.replace("\n", "").strip())
        nps = list([x for x in doc.noun_chunks if x.root.pos_=="NOUN"])
        if self.threshold:
            counts = Counter([x.root.text for x in nps])
            return [np.text for np in nps if counts[np.root.text] >= self.threshold]
            # return [k for k, v in counts.items() if v >= self.threshold]
        return nps

    def get_dbs_ent(self, text: str):
        """ Retrieve entities with Spotlight """
        response = requests.post(
            self.dbpedia_spotlight_api, data=self.get_payload(text=text),
            headers=self.headers, timeout=self.timeout)
        if response.status_code == 200:
            try:
                return set([(resource["@URI"], resource["@surfaceForm"]) \
                            for resource in response.json()["Resources"]])
            except:
                return set()
        print(set)
        return set()

    def get_wordnet_ent(self, text: str):
        words = text.split()
        entities = {"wordnet": []}

        for word in words:
            synsets = wn.synsets(word)
            for synset in synsets:
                entities["wordnet"].append((synset.pos(), synset.name()))

        unique_tuples_set = set(entities["wordnet"])

        found_wordnet_entities_set = set()

        for pos, synset in unique_tuples_set:
            words = [lemma.name() for lemma in wn.synset(synset).lemmas()]
            found_wordnet_entities_set.update(
                {token.text for token in nlp(text) if token.text.lower() in words and token.ent_type_ != ''}
            )

        found_wordnet_entities = list(found_wordnet_entities_set)
        entities['wordnet'] = found_wordnet_entities
        unique_tuples_set.add(tuple(found_wordnet_entities))

        return entities['wordnet']

    def get_spacy_ent(self, text: str):
        doc = nlp(text)
        found_spacy_entities_set = set()

        for ent in doc.ents:
            found_spacy_entities_set.add(ent.text.lower())
        found_spacy_entities_set = list(found_spacy_entities_set)
        return found_spacy_entities_set

    def get_payload(self, text: str):
        """ Payload for requests """
        return {'text': text, 'confidence': self.confidence}

    def __call__(self, text: str):
        """ Extract entities for one string text """
        res = {}
        for option in self.options:
            entities = self.options_to_f[option](text=text)
            res[option] = entities

        return res


if __name__ == '__main__':
    ENTITY_EXTRACTOR = EntityExtractor(options=["dbpedia_spotlight"], confidence=0.35,
                                       db_spotlight_api="http://localhost:2222/rest/annotate",
                                       threshold=1)
    TEXT = """
        The 52-story, 1.7-million-square-foot 7 World Trade Center is a benchmark of innovative design, safety, and sustainability.
        7 WTC has drawn a diverse roster of tenants, including Moody's Corporation, New York Academy of Sciences, Mansueto Ventures, MSCI, and Wilmer Hale.
        The quick brown fox jumps over the lazy dog.
        This is a test sentence without any entities.
        A long list of random words: apple, banana, orange, pineapple, watermelon, kiwi, mango, strawberry, blueberry, raspberry.
        """
    RES = ENTITY_EXTRACTOR(text=TEXT)
    print(RES)
