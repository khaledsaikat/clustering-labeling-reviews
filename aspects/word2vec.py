#!/usr/bin/env python3
import gensim
import numpy as np

"""Load word2vec trained model

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

WORD2VEC_MODEL = None

WORD2VEC_SOURCE_PATH = "/Users/khaled/nltk_data/GoogleNews-vectors-negative300.bin"


def load_word2vec_model():
    """Loading GoogleNews-vectors-negative300 trained model"""
    global WORD2VEC_MODEL
    WORD2VEC_MODEL = gensim.models.KeyedVectors.load_word2vec_format(WORD2VEC_SOURCE_PATH, binary=True)


def similarity(word1, word2):
    """Get similarity score between two words

    :param word1: (string)
    :param word2: (string)
    :return float
    """
    try:
        return WORD2VEC_MODEL.similarity(word1, word2)
    except Exception as e:
        return False


def terms_vectors(terms):
    """Get terms vectors as generator, discard non-existing terms

    :param terms: (list)
    :return generator: vectors
    """
    for term in terms:
        try:
            yield WORD2VEC_MODEL[term]
        except Exception as e:
            pass


def combined_terms_vector(terms):
    """Combine several terms vectors into one

    :param terms: (list)
    :return numpy.ndarray
    """
    terms_vec = terms_vectors(terms)
    combined_vector = np.zeros(300, dtype=np.float32)
    for term_vec in terms_vec:
        combined_vector = combined_vector + term_vec

    return np.array(combined_vector)