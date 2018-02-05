#!/usr/bin/env python3
# Using vector (Doc2Vec) for each sentences, then calculate centroid
##

import operator
import re
from typing import List

import data_loader as dl
import gensim
import numpy as np
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


class TaggedSentencesIter:
    def __init__(self, sentences: List[Sentence]):
        self.sentences = sentences

    def __iter__(self):
        for index, sent in enumerate(self.sentences):
            yield gensim.models.doc2vec.TaggedDocument(WORD_TOKENIZER(sent), [index])


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
    return data["quality"]


def analysis(sentences: List[Sentence], label=None):
    model = gensim.models.Doc2Vec(size=300, min_count=2, iter=1000)
    model.build_vocab(TaggedSentencesIter(sentences))
    ca = ClusterAnalysis(list(model.docvecs))
    distances = ca.sorted_distances
    print(sentences[distances[0][0]])
    # for i in range(10):
    #    print(i, sentences[distances[i][0]], distances[i])
    plt.figure(label)
    ca.boxplot()


def run_all_cluster():
    clusters = dl.loadJsonFromFile("../data/headphone_clusters.json")
    for key, cluster in clusters.items():
        print("___ %s ___" % key)
        analysis(cluster, key)
        print("\n")


def run_single_cluster():
    # sentences = get_20ng_sample()
    sentences = get_review_sample()
    analysis(sentences)


def run_cluster():
    from cluster import AgglomerativeClustering, group_result
    sentences = get_review_sample()
    sentences = ["good sound", "good sound", "bad sound", "bad sound"]
    model = gensim.models.Doc2Vec(size=300, min_count=1, iter=100)
    model.build_vocab(TaggedSentencesIter(sentences))
    X = np.array(list(model.docvecs))
    #X = np.array([(3, 5), (3, 4), (5, 7), (3, 5), (6, 4)])
    ac = AgglomerativeClustering(0.5, "complete", "cosine")
    res = ac.fit_predict(X)
    print(res)
    print(group_result(res))
    print(ac.linkage_matrix)
    ac.dendrogram()


run_cluster()
#run_all_cluster()
# run_single_cluster()
