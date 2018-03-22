#!/usr/bin/env python3
import operator
from typing import List, Callable

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix


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

    def __init__(self, tokens_clusters: List[List[int]] = [[]]):
        self.tokens_clusters = tokens_clusters

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
        indices_group_cols = [self.__sum_columns(X, indices) for indices in self.tokens_clusters]
        for group_index, indices in enumerate(self.tokens_clusters):
            for index in indices:
                X[:, index] = indices_group_cols[group_index]
        # print(__class__, "SummedDFCountTrained:\n", X.toarray())
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
    """Transforming term frequency for test set (clusters)

    target_indices: index of features which have semantic meaning
    similar_columns_bin: Columns of similar tokens as binary value,
    for each row in similer tokens, replace max value with 1 and other values with 0
    """
    target_indices = []
    similar_columns_bin = None

    def __init__(self, tokens_clusters: List[List[int]] = [[]]):
        self.tokens_clusters = tokens_clusters
        self.target_indices = [index for cluster in self.tokens_clusters for index in cluster]

    def fit(self, X, y=None):
        self.similar_columns_bin = self.__build_target_indices_columns(X, self.__transform_row_binary)
        # print(__class__, "similar_columns_bin\n", self.similar_columns_bin.toarray())

    def __build_target_indices_columns(self, X, row_transformer: Callable):
        """Building a column wise matrix for all indices in TOKENS_CLUSTERS"""
        # Making sparse matrix placeholder of (n_sample, n_indices)
        target_indices_columns = lil_matrix((X.shape[0], len(self.target_indices)))
        index_count = 0
        for group_index, indices in enumerate(self.tokens_clusters):
            transformd_columns = self.__manipulate_single_group_columns(X, indices, row_transformer)
            for indices_index, indices_value in enumerate(indices):
                target_indices_columns[:, index_count] = transformd_columns[:, indices_index]
                index_count += 1
        return csr_matrix(target_indices_columns)

    def transform(self, X):
        """Transform columns those have similar tokens

        Replaced by summed value of similar term. column of similar term will look same.
        """
        self.fit(X)
        target_indices_columns = self.__build_target_indices_columns(X, self.__transform_row_summed_zero)
        # print(__class__, "target_indices_columns\n", target_indices_columns.toarray())
        return self.__replace_columns(X, target_indices_columns)

    def transform_same(self, X):
        target_indices_columns = self.__build_target_indices_columns(X, self.__transform_row_max_same)
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
        # print(__class__, "TransformedX:\n", X.toarray())
        return csr_matrix(X)

    def __manipulate_single_group_columns(self, X, indices=[], row_transformer: Callable = None):
        """Transform columns of a single indices group"""
        columns = X[:, indices].toarray()
        if not row_transformer:
            row_transformer: Callable = self.__transform_row
        return csr_matrix([row_transformer(row) for row_index, row in enumerate(columns)])

    def __transform_row(self, row):
        return self.__transform_row_summed_same(row)

    @staticmethod
    def __transform_row_binary(row) -> List:
        """Transform row. Replace max value with 1 and other values with 0

        e.g: Transfor [0, 4, 3] to [0, 1, 0]
        :param row ndarray:
        """
        max_index = max(enumerate(row), key=operator.itemgetter(1))[0]
        bin = 1 if sum(row) else 0
        return [bin if index is max_index else 0 for index, value in enumerate(row)]

    @staticmethod
    def __transform_row_max_same(row) -> List:
        """Transform row. Replace values with max value

        e.g: Transfor [0, 4, 3] to [4, 4, 4], [0, 1 ,0] to [1, 1 ,1], [0, 0 ,0] to [0, 0, 0]
        :param row ndarray:
        """
        max_value = max(row)
        return [max_value for index, value in enumerate(row)]

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
        :param row ndarray:
        """
        row_sum = sum(row)
        return [row_sum for index, value in enumerate(row)]
