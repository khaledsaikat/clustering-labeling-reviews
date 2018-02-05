#!/usr/bin/env python3
import os
from typing import List

import gensim
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

"""Load word2vec trained model

WORD2VEC_SOURCE_PATH can be used as environment variable

For interactive loading load python3 from aspects directory
To Use Word2Vec:
python3
import word2vec as w2v
w2v.load_word2vec_model()
model = w2v.WORD2VEC_MODEL

# Reloading
import imp
def reload():
    imp.reload(w2v)
    w2v.WORD2VEC_MODEL = model
    
reload()
"""

WORD2VEC_MODEL = []

WORD2VEC_SOURCE_PATH = os.environ.get("WORD2VEC_SOURCE_PATH",
                                      "/Users/khaled/nltk_data/GoogleNews-vectors-negative300.bin")


def load_word2vec_model():
    """Loading GoogleNews-vectors-negative300 trained model"""
    global WORD2VEC_MODEL
    WORD2VEC_MODEL = gensim.models.KeyedVectors.load_word2vec_format(WORD2VEC_SOURCE_PATH, binary=True)


def similarity(sentence_word1: str, sentence_word2: str) -> float:
    """Get similarity score between two words or sentences

    :param sentence_word1: (string)
    :param sentence_word2: (string)
    :return float
    """
    return cosine_similarity([mean_terms_vector(sentence_word1)], [mean_terms_vector(sentence_word2)])[0][0]


def terms_vectors(terms: List[str]):
    """Get terms vectors as generator, discard non-existing terms

    :param terms: (list)
    :return generator: vectors
    """
    return [WORD2VEC_MODEL[term] for term in terms if term in WORD2VEC_MODEL]


def valid_words(terms: List[str]):
    return [term for term in terms if term in WORD2VEC_MODEL]


def combined_terms_vector(terms: List[str]):
    """Combine several terms vectors into one
    @deprecated use sum_terms_vector() instead

    :param terms: (list)
    :return numpy.ndarray
    """
    terms_vec = terms_vectors(terms)
    combined_vector = np.zeros(300, dtype=np.float32)
    for term_vec in terms_vec:
        combined_vector = combined_vector + term_vec

    return np.array(combined_vector)


def sum_terms_vector(terms):
    """
    Summing terms vector together
    :param terms: List or string
    :return:
    """

    terms = terms.split() if type(terms) is str else terms
    sumed_vectors = np.zeros(300, dtype=np.float32)
    for term_vec in terms_vectors(terms):
        sumed_vectors = sumed_vectors + term_vec

    return sumed_vectors


def mean_terms_vector(sentence: str):
    """Get a term vector from sentence with averaging values"""
    vectors = list(terms_vectors(sentence.split()))
    # return np.mean(vectors, axis=0) if vectors else 0
    return np.mean(vectors, axis=0) if vectors else np.zeros(300, dtype=np.float32)
