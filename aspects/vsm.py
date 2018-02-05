#!/usr/bin/env python3
import operator
from typing import List, Tuple

import numpy as np
import word2vec as w2v
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import make_pipeline
from pprint import pprint

"""
cats = ["sci.space"]
from sklearn.datasets import fetch_20newsgroups
newsgroups = fetch_20newsgroups(categories=cats)


words = ['noise', 'sound', 'loud', 'cancel', 'canceling', 'cancellation', 'headphones', 'earbuds', 'earbud', 'ipod',
        'cord', 'cable', 'cords', 'cables', 'jacket', 'cheap', 'price', 'color', 'plastic', 'tangle']

"""

corpus = [
    'This is the first document.',
    'This is the second second document.',
    'And the third one.',
    'Is this the first document?',
]

VECTORIZER = None


def init_vectorizer(documents):
    global VECTORIZER
    VECTORIZER = TfidfVectorizer(stop_words="english", ngram_range=(1, 1))
    VECTORIZER.fit(documents)
    VECTORIZER.w2v_features_ = []


class BaseEstimator:
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X


class W2VTransformer(BaseEstimator):
    def __init__(self, multiply_scalar=False):
        self.init_w2v_features()
        self.multiply_scalar = multiply_scalar

    def transform(self, X):
        """Transforming X into w2v features

        Transforming (n_samples, n_features) into (n_samples, n_features, 300)
        Multiply with scalar if multiply_scalar is true otherwise use  direct w2v vector or zero vector
        """
        if self.multiply_scalar:
            X = np.array([[self.w2v_features[index] * scalar_value for index, scalar_value in enumerate(row)] for row in
                          X.toarray()])
        else:
            X = np.array(
                [[self.w2v_features[index] * (1 if scalar_value > 0 else 0) for index, scalar_value in enumerate(row)]
                 for row in
                 X.toarray()])
        return X

    def init_w2v_features(self):
        global VECTORIZER
        if not VECTORIZER.w2v_features_:
            VECTORIZER.w2v_features_ = list(self.generate_w2v_features())

    @staticmethod
    def generate_w2v_features():
        for feature_name in VECTORIZER.get_feature_names():
            yield w2v.mean_terms_vector(feature_name)

    @property
    def w2v_features(self):
        return VECTORIZER.w2v_features_


class TermLabeler(BaseEstimator):
    def __int__(self):
        pass

    def fit_predict(self, X, y=None):
        return [self._sort_from_max(row) for row in X.toarray()]

    @staticmethod
    def _sort_from_max(row: List[float]) -> List[Tuple[int, float]]:
        """Sort list from max to min with index"""
        return sorted(enumerate(row), key=operator.itemgetter(1), reverse=True)


def inspect_labels(results, n_tokens=10):
    """Convert term index to terms"""
    return [
        [(VECTORIZER.get_feature_names()[value_tuple[0]], value_tuple[1]) for tuple_index, value_tuple in enumerate(row)
         if tuple_index < n_tokens] for row in results]


class VSM:
    def __init__(self, documents: List[str]):
        """documents: List of docs from all clusters"""
        self.init_vectorizer(documents)
        self.init_w2v_features()

    @staticmethod
    def init_vectorizer(documents: List[str]):
        global VECTORIZER
        VECTORIZER = TfidfVectorizer(stop_words="english", ngram_range=(1, 1))
        VECTORIZER.fit(documents)

    def init_w2v_features(self):
        global VECTORIZER
        VECTORIZER.w2v_features_ = list(self.generate_w2v_features())

    @staticmethod
    def generate_w2v_features():
        for feature_name in VECTORIZER.get_feature_names():
            yield w2v.mean_terms_vector(feature_name)

    @property
    def idf_vector(self):
        return VECTORIZER.idf_

    @property
    def w2v_features(self):
        return VECTORIZER.w2v_features_

    @property
    def feature_vector(self):
        return VECTORIZER.get_feature_names()

    def transform(self, documents: List[str]):
        tfidf_vector = VECTORIZER.transform(documents)
        return np.array([[self.w2v_features[index] * tfidf_value for index, tfidf_value in enumerate(row)] for row in
                         tfidf_vector.toarray()])

    @staticmethod
    def example():
        model = VSM(corpus)
        print(model.feature_vector)
        print(model.idf_vector)
        print(VECTORIZER.transform([corpus[1]]).toarray())
        print(list(model.transform([corpus[1]])))


class ClusterAnalysis:
    def __init__(self, documents: List[str]):
        vectorizer = VSM(documents)

        # Shape: (n_sample, n_feature, 300)
        # each sample (n_feature, 300)
        self.vectors: np.ndarray = vectorizer.transform(documents)

        # Shape: (n_feature, 300)
        self.center: np.ndarray = np.mean(self.vectors, axis=0)

    def distance(self, vector: np.ndarray) -> float:
        """Distance from center vector"""
        return sum(sum((self.center - vector) ** 2))

    def show_distances(self):
        weights = [self.distance(vec) for vec in self.vectors]
        print(weights)
        return weights

    def get_min_index_distance(self) -> Tuple[int, float]:
        """Get minimum distance from center (index, distance)"""
        weights = [self.distance(vec) for vec in self.vectors]
        return min(enumerate(weights), key=operator.itemgetter(1))

    def distance_sum(self) -> float:
        """Sum of all distance from center"""
        return sum([self.distance(vec) for vec in self.vectors])

    def test(self, index):
        print(cosine_similarity(self.center, self.vectors[index]).shape)


def example():
    ca = ClusterAnalysis(corpus)
    ca.show_distances()
    print(ca.get_min_index_distance())
    print(ca.distance_sum())


def run(documents: List[str]):
    ca = ClusterAnalysis(documents)
    d = ca.show_distances()
    index, min_d = ca.get_min_index_distance()
    print(index, min_d)
    print(ca.distance_sum())
    print(ca.distance_sum() / len(d))
    # print(ca.test(index))
    plt.boxplot(d)
    plt.show()


def run_cluster_sample():
    import data_loader as dl
    data = dl.loadJsonFromFile("../data/headphone_clusters.json")

    # all documents
    docs = set(doc for cluster in data.values() for doc in cluster)
    # cluster as whole text
    clusters_text = (" ".join(doc for doc in cluster) for cluster in data.values())

    init_vectorizer(docs)
    X = VECTORIZER.transform(clusters_text)
    pipeline = make_pipeline(TermLabeler())
    result = pipeline.fit_predict(X)
    pprint(inspect_labels(result, 3))


def test():
    """
    VSM.example()
    print(ClusterAnalysis(corpus).distance_sum())

    init_vectorizer(corpus)
    X = VECTORIZER.transform(corpus)
    print(X.toarray())

    pipeline = make_pipeline(BaseEstimator(), W2VTransformer())
    print(pipeline.transform(X))
    """

    init_vectorizer(corpus)
    X = VECTORIZER.transform(corpus)
    pipeline = make_pipeline(TermLabeler())
    r = pipeline.fit_predict(X)
    print(r)
    print(inspect_labels(r))

if __name__ == "__main__":
    run_cluster_sample()

