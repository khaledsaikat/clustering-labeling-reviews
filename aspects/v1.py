#!/usr/bin/env python3
import operator
from pprint import pprint
from typing import List, Tuple

import numpy as np
import word2vec as w2v
from cluster import AgglomerativeClustering, group_result
from matplotlib import pyplot as plt
from nltk.util import bigrams
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
import operator
from nltk.tokenize import sent_tokenize

"""
TODO: Cluster tokens for bigram and trigram
"""

VECTORIZER = None

TOKENS_CLUSTERS: List[List[int]] = []


def init_vectorizer(documents, init_w2v_features=False):
    global VECTORIZER
    #VECTORIZER = CountVectorizer(stop_words="english", ngram_range=(1, 3), binary=True)
    VECTORIZER = CountVectorizer(stop_words="english", min_df=2, max_df=0.5, ngram_range=(1, 3))
    VECTORIZER.fit(documents)
    tokens_clusters()
    if init_w2v_features:
        VECTORIZER.w2v_features_ = np.array(
            [w2v.mean_terms_vector(feature_name) for feature_name in VECTORIZER.get_feature_names()])


class BaseEstimator:
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


class SimilarDFTransformer(BaseEstimator):
    """Transforming document frequency for training set"""

    def transform(self, X):
        """Combine document frequency of similar terms by summing them.

        Replace column for similar tokens with their summed column
        Both columns will look same after replacement.
        :param X: Expect X from CountVectorizer
        """
        return self.__transform_columns_summed_same(X)

    def __transform_columns_summed_same(self, X: csr_matrix):
        """Transform columns. Replace column with summed column. All columns in a group will look same"""
        X = lil_matrix(X)
        indices_group_cols = [self.__sum_columns(X, indices) for indices in TOKENS_CLUSTERS]
        for group_index, indices in enumerate(TOKENS_CLUSTERS):
            for index in indices:
                X[:, index] = indices_group_cols[group_index]
        print(__class__, "SummedDFCountTrained:\n", X.toarray())
        return csr_matrix(X)

    @staticmethod
    def __sum_columns(X: csr_matrix, indices: List[int] = []) -> csr_matrix:
        """Summing multiple columns to a single column

        :param X: csr_matrix with shape of (n_samples, n_features)
        :param indices: list of index for columns those are going to summed
        :return csr_matrix: shape is (n_samples, 1), single column with n_samples rows
        """
        n_samples, n_features = X.shape
        summed = csr_matrix(np.zeros(n_samples).reshape(n_samples, 1))
        for index in indices:
            summed += X[:, index]

        return summed


class SimilarTFTransformer(BaseEstimator):
    """Transforming term frequency for test set"""
    target_indices = []
    similar_columns_bin = None

    def __init__(self):
        self.target_indices = [index for cluster in TOKENS_CLUSTERS for index in cluster]

    def fit(self, X, y=None):
        self.similar_columns_bin = self.__build_target_indices_columns(X, self.__transform_row_binary)
        print(__class__, "similar_columns_bin\n", self.similar_columns_bin.toarray())

    def __build_target_indices_columns(self, X, row_transformer=None):
        """Building a column wise matrix for all indices in TOKENS_CLUSTERS"""
        # Making sparse matrix placeholder of (n_sample, n_indices)
        target_indices_columns = lil_matrix((X.shape[0], len(self.target_indices)))
        index_count = 0
        for group_index, indices in enumerate(TOKENS_CLUSTERS):
            transformd_columns = self.__manipulate_single_group_columns(X, indices, row_transformer)
            for indices_index, indices_value in enumerate(indices):
                target_indices_columns[:, index_count] = transformd_columns[:, indices_index]
                index_count += 1
        return csr_matrix(target_indices_columns)

    def transform(self, X):
        """Transform columns those have similar tokens"""
        target_indices_columns = self.__build_target_indices_columns(X, self.__transform_row_summed_same)
        print(__class__, "target_indices_columns\n", target_indices_columns.toarray())
        return self.__replace_columns(X, target_indices_columns)

    def filter_similar(self, X):
        """Filter similar tokens with assigning zero value"""
        X = lil_matrix(X)
        for index, vocabulary_index in enumerate(self.target_indices):
            # Converted into toarray() because we need pointwise multiplication
            # Multiplying a column with a binary column
            X[:, vocabulary_index] = X[:, vocabulary_index].toarray() * self.similar_columns_bin[:, index].toarray()
        return csr_matrix(X)

    def __replace_columns(self, X, indices_columns):
        """Replace columns from X with indices_columns"""
        X = lil_matrix(X)
        for index, vocabulary_index in enumerate(self.target_indices):
            X[:, vocabulary_index] = indices_columns[:, index]
        print(__class__, "TransformedX:\n", X.toarray())
        return csr_matrix(X)

    def __manipulate_single_group_columns(self, X, indices=[], row_transformer=None):
        """Transform columns of a single indices group"""
        columns = X[:, indices].toarray()
        if not row_transformer:
            row_transformer = self.__transform_row
        return csr_matrix([row_transformer(row) for row_index, row in enumerate(columns)])

    def __transform_row(self, row):
        return self.__transform_row_summed_same(row)

    @staticmethod
    def __transform_row_binary(row) -> List:
        """Transform row. Replace max value with 1 value and other values with zero

        e.g: Transfor [0, 4, 3] to [0, 1, 0]
        :param row ndarray:
        """
        max_index = max(enumerate(row), key=operator.itemgetter(1))[0]
        bin = 1 if sum(row) else 0
        return [bin if index is max_index else 0 for index, value in enumerate(row)]

    @staticmethod
    def __transform_row_summed_zero(row) -> List:
        """Transform row. Replace max value with summed value and other values with zero

        e.g: Transfor [0, 4, 3] to [0, 7, 0]
        :param row ndarray:
        """
        max_index = max(enumerate(row), key=operator.itemgetter(1))[0]
        return [sum(row) if index is max_index else 0 for index, value in enumerate(row)]

    @staticmethod
    def __transform_row_summed_same(row) -> List:
        """Transform row. Replace values with summed value

        e.g: Transfor [0, 4, 3] to [7, 7, 7]
        """
        row_sum = sum(row)
        return [row_sum for index, value in enumerate(row)]


class CombineGramTransformer(BaseEstimator):
    def __init__(self):
        self.feature_names: List = VECTORIZER.get_feature_names()

    def transform(self, X):
        X = lil_matrix(X)
        for row_index, row_csr in enumerate(X):
            row = row_csr.toarray()[0]
            X[row_index] = lil_matrix([self._combine_term_weight(row, index, value) for index, value in enumerate(row)])
        return csr_matrix(X)

    def _combine_term_weight(self, row, feature_index, feature_weight):
        if feature_weight == 0:
            return 0
        feature_name = self.feature_names[feature_index]
        if len(feature_name.split()) == 1:
            return feature_weight
        elif len(feature_name.split()) == 2:
            return self._combine_bigrams(row, feature_name) + feature_weight
        elif len(feature_name.split()) == 3:
            return self._combine_trigrams(row, feature_name) + feature_weight

    @staticmethod
    def _combine_bigrams(row, feature_name: str):
        """Calculate result of two unigram as bigram

        :param feature_name: bigram feature
        """
        return sum(row[VECTORIZER.vocabulary_[name]] if name in VECTORIZER.vocabulary_ else 0 for name in
                   feature_name.split()) / 2

    def _combine_trigrams(self, row, feature_name: str):
        """Calculate result of two bigram as trigram"""
        tokens = [" ".join(v) for v in bigrams(feature_name.split())]
        return sum([self._combine_bigrams(row, token) for token in tokens]) / 2


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
            VECTORIZER.w2v_features_ = np.array(self.generate_w2v_features())

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
        """Sort list from max to min with index List[(index, value)]"""
        return sorted(enumerate(row), key=operator.itemgetter(1), reverse=True)


def inspect_labels(results, n_tokens=10):
    """Convert term index to terms"""
    return [
        [(VECTORIZER.get_feature_names()[value_tuple[0]], value_tuple[1]) for tuple_index, value_tuple in enumerate(row)
         if tuple_index < n_tokens] for row in results]


def tokens_clusters():
    """Cluster tokens

    Return indices of tokens in clusters
    """
    global TOKENS_CLUSTERS
    ac = AgglomerativeClustering(0.5, "complete", "cosine")
    valid_words = w2v.valid_words(VECTORIZER.get_feature_names())
    result = ac.fit_predict(w2v.terms_vectors(valid_words))
    clusters = group_result(result, 2)
    TOKENS_CLUSTERS = [[VECTORIZER.vocabulary_[valid_words[word_index]] for word_index in cluster] for cluster in
                       clusters]


def example():
    ca = ClusterAnalysis(corpus_train_samples)
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
    docs = list(set(doc for cluster in data.values() for doc in cluster))
    # cluster as whole text
    clusters_text = [" ".join(doc for doc in cluster) for cluster in data.values()]
    sent_clusters = [cluster for cluster in data.values()]
    run_analysis(docs, clusters_text, sent_clusters)


def run_20ng_sample():
    categories = ["alt.atheism", "soc.religion.christian", "comp.graphics", "sci.med"]
    clusters = [fetch_20newsgroups(subset="train", categories=[cat]).data[:50] for cat in categories]
    docs = list(set(doc for cluster in clusters for doc in cluster))
    clusters_text = [" ".join(doc for doc in cluster) for cluster in clusters]
    sent_clusters = [sent_tokenize(str(cluster)) for cluster in clusters]
    run_analysis(docs, clusters_text, sent_clusters)


def run_analysis(corpus_train: List[str], corpus_test: List[str], sent_clusters: List[List[str]]=[]):
    # Traning
    init_vectorizer(corpus_train)
    X = VECTORIZER.transform(corpus_train)
    print("FeatureNamesCount:", len(VECTORIZER.get_feature_names()))
    print("FeatureNames:", VECTORIZER.get_feature_names()[:10])
    print("TOKENS_CLUSTERS:", TOKENS_CLUSTERS[:10])
    print("TOKENS_CLUSTERS:",
          [[VECTORIZER.get_feature_names()[i] for i in cluster] for cluster in TOKENS_CLUSTERS][:10])
    print("VECTORIZER X\n:", X.toarray())
    X = SimilarDFTransformer().transform(X)
    tfidf = TfidfTransformer().fit(X)
    # print("IDF:", tfidf.idf_)

    # Apply
    X = VECTORIZER.transform(corpus_test)
    print("TestVECTORIZER\n", X.toarray())
    stft = SimilarTFTransformer()
    X = stft.fit_transform(X)
    X = tfidf.transform(X)
    # print("TfIdf:", X.toarray())

    pipeline = make_pipeline(CombineGramTransformer())
    X = pipeline.transform(X)
    X = stft.filter_similar(X)

    # Labeling
    labeler = TermLabeler()
    result = labeler.fit_predict(X)
    pprint(inspect_labels(result, 5))

    print(X.toarray())
    print(X.shape)

    # Sent Labeling
    for cluster_index, sent_cluster in enumerate(sent_clusters):
        Y = VECTORIZER.transform(sent_cluster)
        #print(Y.toarray())
        sent_weights = []
        for sent_index, sent in enumerate(sent_cluster):
            v = Y[sent_index].multiply(X[cluster_index])
            sent_weights.append(v.sum()/len(sent))
        #print(sent_weights)
        max_index = max(enumerate(sent_weights), key=operator.itemgetter(1))[0]
        print(sent_cluster[max_index], "\n")





def run_samples():
    corpus_train_samples = [
        'This is the first document in collection.',
        'This is the second second document documents.',
        'And the third one in collection color.',
        'Is this the first document colors?',
    ]

    corpus_test_samples = [
        "another test document color colors",
        "test document for second time document collection",
        "This is a final one"
    ]

    sent_clusters = [corpus_test_samples, corpus_test_samples, corpus_test_samples]
    run_analysis(corpus_train_samples, corpus_test_samples, sent_clusters)


if __name__ == "__main__":
    run_samples()
