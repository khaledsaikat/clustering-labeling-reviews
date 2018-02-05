#!/usr/bin/env python3

###
# Using vector for each sentences, then calculate centroid
# sentences_vector = sum(words_vectors)  (w2v)
##
"""
python3
import word2vec as w2v
w2v.load_word2vec_model()
model = w2v.WORD2VEC_MODEL

import v4
v4.run()

from imp import reload
reload(v4)
"""

import operator
import re
from typing import List

import data_loader as dl
import numpy as np
import word2vec as w2v
from matplotlib import pyplot as plt
from nltk.tokenize import sent_tokenize
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

Word = str
Sentence = str

WORD_TOKENIZER = CountVectorizer(stop_words="english").build_analyzer()


class ClusterAnalysis:
    """Analysis of single cluster"""
    center: np.ndarray = None
    distances: List[float]

    def __init__(self, vectors: List[np.ndarray]):
        self.vectors = vectors
        self.__set_center()
        self.__calculate_all_distances()

    def __set_center(self):
        self.center = np.mean(self.vectors, axis=0)

    def __distance(self, vector: np.ndarray) -> float:
        """Similarity from center vector"""
        return cosine_similarity([self.center], [vector])[0][0]

    def __calculate_all_distances(self):
        self.distances = [self.__distance(vec) for vec in self.vectors]

    @property
    def sorted_distances(self) -> List[float]:
        """Get sorted distances"""
        return sorted(enumerate(self.distances), key=operator.itemgetter(1), reverse=True)

    @property
    def average_distance(self):
        return sum(self.distances) / len(self.distances)

    def boxplot(self):
        plt.boxplot(self.distances)
        plt.show()


def get_20ng_sample():
    categories = ["soc.religion.christian"]
    documents = fetch_20newsgroups(subset="train", categories=[categories[0]]).data[:50]
    documents = [_cleanup_20ng_doc(doc) for doc in documents]
    documents = [sent_tokenize(doc) for doc in documents]
    all_sentences = [sent for doc in documents for sent in doc]
    return all_sentences


def _cleanup_20ng_doc(document: str):
    document = re.sub("From:.+\n", "", document)
    document = re.sub("Lines:.+\n", "", document)
    document = document.replace("\n", " ")
    return document.lower()


def get_review_sample() -> List[Sentence]:
    data = dl.loadJsonFromFile("../data/headphone_clusters.json")
    return data["sound quality"]


def get_raw_sentences() -> List[Sentence]:
    """Return list of raw sentences"""
    return get_review_sample()
    # return get_20ng_sample()


def get_tokenized_sentences(sentences: List[Sentence]) -> List[List[Word]]:
    tokenized_sentences = [WORD_TOKENIZER(sent) for sent in sentences]
    return tokenized_sentences
    # return [sent for sent in tokenized_sentences if len(sent) >= 3]


def analysis(sentences: List[Sentence]):
    tokenized_sentences: List[List[Word]] = get_tokenized_sentences(sentences)
    sentences_vectors = [w2v.sum_terms_vector(" ".join(sent)) for sent in tokenized_sentences]
    ca = ClusterAnalysis(sentences_vectors)
    distances = ca.sorted_distances
    for i in range(3):
        print(i, sentences[distances[i][0]])
    # ca.boxplot()


def run_all_cluster():
    clusters = dl.loadJsonFromFile("../data/headphone_clusters.json")
    for key, cluster in clusters.items():
        print("___ %s ___" % key)
        analysis(cluster)
        print("\n")


def run_single_cluster():
    # sentences = get_20ng_sample()
    sentences = get_review_sample()
    analysis(sentences)


def run_textrank():
    #from gensim.summarization import keywords
    from summa.keywords import keywords

    clusters = dl.loadJsonFromFile("../data/headphone_clusters.json")
    for key, cluster in clusters.items():
        print("___ %s ___" % key)
        print(keywords( ". ".join(cluster), words=10, scores=False))
        print("\n")


if __name__ == "__main__":
    run_textrank()
