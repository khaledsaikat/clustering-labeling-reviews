#!/usr/bin/env python3
import operator

import data_loader as dl
import numpy as np
import word2vec as w2v
from nltk.corpus import stopwords
from textblob import TextBlob

"""Find clusters labels by using centroid

For interactive loading, load python3 from aspects directory
```
import word2vec as w2v
import centroid_labels as cl
w2v.load_word2vec_model()
cl.inspect_labels()
```
"""


class NormalizeClusters:
    """Normalize all clusters"""

    clusters = []

    def __init__(self, clusters):
        """Contains clusters as a list. Each cluster contains list of docs"""
        self.clusters = clusters
        self.normalize()

    def normalize(self):
        """Normalize each doc in cluster"""
        self.clusters = [[self._normalize(doc) for doc in cluster] for cluster in self.clusters]

    def _normalize(self, doc):
        """Lower text, and filter stop words"""
        doc = TextBlob(doc)
        doc = doc.lower()
        # doc = doc.correct()
        doc = self._filter_stop_words(doc)
        return doc

    @staticmethod
    def _filter_stop_words(doc):
        words = [word for word in doc.words if not word in stopwords.words("english")]
        return TextBlob(" ".join(words))


class CentroidLabel:
    cluster = []

    # 300 dimension numpy.ndarray
    docs_vector = []

    center = []

    def __init__(self, cluster):
        self.cluster = cluster
        self.doc_vector()
        self.calculate_centroid()

    def doc_vector(self):
        self.docs_vector = np.array([w2v.combined_terms_vector(doc.words) for doc in self.cluster])

    def calculate_centroid(self):
        """calculate the centroid"""
        length, dimension = self.docs_vector.shape
        self.center = np.array([np.sum(self.docs_vector[:, i]) / length for i in range(dimension)])

    @staticmethod
    def distance(v1, v2):
        return sum((v1 - v2) ** 2)

    def get_closest_doc_index(self):
        weights = [self.distance(self.center, vec) for vec in self.docs_vector]
        return min(enumerate(weights), key=operator.itemgetter(1))[0]


def inspect_labels():
    data = {
        "a": ["I'm a  sound bit, of a headphone/earphone snob.",
              "these are super cheap. and mostly you get what you pay for."],
        "b": ["Unbelievable sound for the price", "they sound great and are built well."]
    }
    data = dl.loadJsonFromFile("../data/headphone_clusters.json")

    pre_labels = [k for k, v in data.items()]
    raw_clusters = [v for k, v in data.items()]

    clusters = NormalizeClusters(raw_clusters).clusters
    for index, cluster in enumerate(clusters):
        cl = CentroidLabel(cluster)
        label = raw_clusters[index][cl.get_closest_doc_index()]
        print(pre_labels[index], ":", label)


if __name__ == "__main__":
    w2v.load_word2vec_model()
    inspect_labels()
