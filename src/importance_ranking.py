import os
import spacy
import networkx as nx
import numpy as np
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from summa import summarizer
from typing import Union, List
from loguru import logger


class ImportanceRanker:
    def __init__(self, ranking: str, int_threshold: Union[int, None] = None,
                 perc_threshold: Union[float, None] = None,
                 word2vec_model_path: Union[str, None] = None):
        self.options_ranker = ["page_rank", "text_rank", "tfidf","word2vec"]
        self.word2vec_model_path = word2vec_model_path if word2vec_model_path else "word2vec.model"
        self.options_to_f = {
            "page_rank": self.compute_page_rank,
            "text_rank": self.compute_text_rank,
            "tfidf": self.tfidf_importance_ranking,
            "word2vec": self.word_embedding_similarity,
        }
        self.params = {
            "ranking": ranking,
        }
        self.ranking = ranking
        self.int_threshold = int_threshold
        self.perc_threshold = perc_threshold
        self.nlp = spacy.load("en_core_web_lg")
        

    def check_params(self, ranking, int_threshold, perc_threshold):
        if ranking not in self.options_ranker:
            raise ValueError(f"`ranking` param should be in {self.options_ranker}")
        if (int_threshold and perc_threshold) or ((not int_threshold) and (not perc_threshold)):
            raise ValueError("Either `int_threshold` or `perc_threshold` should be non-null (only one)")
        if int_threshold:
            if not isinstance(int_threshold, int):
                raise ValueError("`int_threshold` should be int")
        if perc_threshold:
            if (not isinstance(perc_threshold, float)) or  not (0 < perc_threshold < 1):
                raise ValueError("`perc_threshold` should be a float between 0 and 1")

    def compute_page_rank(self, sentences):
        """Compute the importance ranking of a list of sentences based on page rank"""
        if len(sentences) == 0:
            return []

        vectorizer = CountVectorizer()
        sentence_vectors = vectorizer.fit_transform(sentences)

        if len(vectorizer.vocabulary_) == 0:
            return []

        similarity_matrix = cosine_similarity(sentence_vectors)
        graph = nx.from_numpy_array(similarity_matrix)
        scores = nx.pagerank(graph)
        ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
        return [x[1] for x in ranked_sentences]

    def tfidf_importance_ranking(self, sentences):
        """Compute the importance ranking of a list of sentences based on tf-idf embedding"""
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(sentences)
        tfidf_scores = tfidf_matrix.sum(axis=1).A1
        ranked_indices = tfidf_scores.argsort()[::-1]
        ranked_sentences = [sentences[i] for i in ranked_indices]
        return ranked_sentences

    def compute_text_rank(self, sentences):
        """Using the Summa library --> be careful sometimes return empty summary"""
        text = ' '.join(sentences)
        summary = summarizer.summarize(text)
        sentences = summary.split('.')
        sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
        return sentences

    def train_word2vec_model(self, sentences):
        """Train and save a Word2Vec model"""
        model = Word2Vec(sentences=sentences, vector_size=100, window=5, min_count=1, workers=4)
        # model.save(self.word2vec_model_path)
        return model

    def load_word2vec_model(self, word2vec_file):
        """Load Word2Vec model from file"""
        return Word2Vec.load(word2vec_file)

    def average_embedding(self, sentence, model, dim=100):
        """Compute the average word embedding for a sentence"""
        embedding = np.zeros(dim)
        count = 0
        for word in sentence:
            if word in model.wv:
                embedding += model.wv[word]
                count += 1
        if count > 0:
            embedding /= count
        return embedding

    def word_embedding_similarity(self, sentences):
        """Compute the importance ranking of a list of sentences based on Word2Vec embeddings"""

        # transforming sentences into list of list of tokens
        sentences_tokens = [self.nlp(sent) for sent in sentences]
        sentences_tokens = [[x.text for x in doc] for doc in sentences_tokens]

        if not os.path.exists(self.word2vec_model_path):
            word2vec_model = self.train_word2vec_model(sentences=sentences_tokens)
        else:
            word2vec_model = self.load_word2vec_model(self.word2vec_model_path)
        sentence_embeddings = [self.average_embedding(sentence, word2vec_model) for sentence in sentences_tokens]
        similarity_matrix = cosine_similarity(sentence_embeddings)
        importance_scores = np.sum(similarity_matrix, axis=1)
        ranked_indices = importance_scores.argsort()[::-1]
        ranked_sentences = [sentences[i] for i in ranked_indices]
        return ranked_sentences
    
    def __call__(self, sentences: List[str]):
        if not isinstance(sentences, list):
            raise ValueError("Input must be a list of sentences")
        ranked_sent = self.options_to_f[self.ranking](sentences)
        if not ranked_sent:
            ranked_sent = sentences

        if self.int_threshold:
            limit = min(self.int_threshold, len(ranked_sent))
            ranked_sent = ranked_sent[:limit]
        if self.perc_threshold:
            limit = int(self.perc_threshold * len(ranked_sent))
            ranked_sent = ranked_sent[:limit]

        return ranked_sent


if __name__ == '__main__':
    sentences = [
        "Automatic summarization is the process of reducing a text document with a computer program in order to create a summary that retains the most important points of the original document",
        "As the problem of information overload has grown, and as the quantity of data has increased, so has interest in automatic summarization",
        "Technologies that can make a coherent summary take into account variables such as length, writing style and syntax",
    ]
    # ["page_rank", "text_rank", "tfidf","word2vec"]
    ranker = ImportanceRanker(ranking="page_rank")
    ranked_page_rank = ranker(sentences)
    print(f"PAGE RANK\n{ranked_page_rank}\n==========")

    ranker = ImportanceRanker(ranking="tfidf")
    ranked_tfidf = ranker(sentences)
    print(f"TFIDF\n{ranked_tfidf}\n==========")

    ranker = ImportanceRanker(ranking="text_rank")
    ranked_text_rank = ranker(sentences)
    print(f"TEXT RANK\n{ranked_text_rank}\n==========")

    ranker = ImportanceRanker(ranking="word2vec")
    ranked_word2vec = ranker(sentences)
    print(f"WORD2VEC\n{ranked_word2vec}\n==========")