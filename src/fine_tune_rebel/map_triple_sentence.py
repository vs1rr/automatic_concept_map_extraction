# -*- coding: utf-8 -*-
"""

"""
import os
import spacy
from datetime import datetime
import pandas as pd

NLP = spacy.load("en_core_web_lg")

def find_sent(triple, sents):
    """ Find original sentence from which triple was extracted """
    nbs = [len(sents)]
    cands = [sent for sent in sents if triple[0].text in sent.text.lower()]
    nbs.append(len(cands))
    cands = [sent for sent in cands if triple[2].text in sent.text.lower()]
    nbs.append(len(cands))

    for token in [tok for tok in triple[1] if tok.pos_ == "VERB"]:
        cands = [sent for sent in cands if token.lemma_ in [t.lemma_ for t in sent]]
    nbs.append(len(cands))
    return nbs, cands

if __name__ == '__main__':
    T_START = datetime.now()
    NB_TRIPLES, NB_TRIPLES_HYPER, NB_TRIPLES_HYPO,  NB_TRIPLES_MAPPED = 0, 0, 0, 0
    CONTEXT, TRIPLETS = [], []

    for topic in sorted(os.listdir("src/fine_tune_rebel/Corpora/Biology/")):
        print(topic)
        
        try:
            file_n = os.listdir(f"src/fine_tune_rebel/Corpora/Biology/{topic}/gold/")[0]
            triples = pd.read_csv(f"src/fine_tune_rebel/Corpora/Biology/{topic}/gold/{file_n}", header=None)
            file_n = os.listdir(f"src/fine_tune_rebel/Corpora/Biology/{topic}/text/")[0]
            text = open(f"src/fine_tune_rebel/Corpora/Biology/{topic}/text/{file_n}", 'r').read()
            doc = NLP(text)
            sents = list(doc.sents)

            for i in range(triples.shape[0]):
                NB_TRIPLES += 1
                TRIPLE = [NLP(elt) for elt in list(triples.iloc[i])[0].split("\t")]
                if TRIPLE[1].text == "$HYPONYM":
                    NB_TRIPLES_HYPO += 1
                if TRIPLE[1].text == "$HYPERNYM":
                    NB_TRIPLES_HYPER += 1
                NBS, CANDS = find_sent(triple=TRIPLE, sents=sents)
                if NBS[-1] == 1 and TRIPLE[1].text not in ["$HYPONYM", "$HYPERNYM"]:
                    NB_TRIPLES_MAPPED += 1

                    CONTEXT.append(CANDS[-1].text)
                    TRIPLETS.append(f"<triplet> {TRIPLE[0].text} <subj> {TRIPLE[2].text} <obj> {TRIPLE[1].text}")
                print(f"Triple {i}/{triples.shape[0]}", "\t", NBS)
        except Exception as e:
            print(e)
        print("===============")

    print(NB_TRIPLES, NB_TRIPLES_HYPER, NB_TRIPLES_HYPO, NB_TRIPLES_MAPPED)
    pd.DataFrame({"context": CONTEXT, "triplets": TRIPLETS}).to_csv("src/fine_tune_rebel/cm_biology.csv")
    T_END = datetime.now()
    print(f"Finished at {T_END}, took {T_END-T_START}")
