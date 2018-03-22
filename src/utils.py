import math
from collections import Counter
from operator import itemgetter
from typing import Tuple, Union, List

import nltk
import numpy as np
import spacy
from gensim.parsing.porter import PorterStemmer
from gensim.parsing.preprocessing import STOPWORDS
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from textblob import TextBlob

Word = str
Sentence = str
Term = Union[Word, Tuple[Word, Word], Tuple[Word, Word, Word]]
Document = TextBlob

WORD_TOKENIZER = CountVectorizer(stop_words=STOPWORDS).build_analyzer()

REGEXP_TOKENIZER = RegexpTokenizer("\w+|\$[\d\.]+|\S+")

POTER_STEMMER = PorterStemmer()

NLP = spacy.load("en")


def spacy_tokenizer(text: str, accepted_tags=["NN", "JJ", "VB"]):
    doc = NLP(text)
    # tokens = [token for token in doc if not token.text.lower() in STOPWORDS]

    words = [token.text.lower() for token in doc if len(token.tag_) > 1 and token.tag_[:2] in accepted_tags]
    return [word for word in words if not word in STOPWORDS]


def raw_word_tokenizer(text: str) -> List[Word]:
    """Only tokenizer word without changing anything"""
    return REGEXP_TOKENIZER.tokenize(text)


# def pos_filtered_tokenizer(text: Sentence, accepted_tags=["NN", "JJ", "VB"]) -> List[str]:
#    return word_tokenizer(" ".join(_pos_filtered_tokenizer(Sentence, accepted_tags)))


def pos_filtered_tokenizer(text: Sentence, accepted_tags=["NN", "JJ", "VB"]) -> List[str]:
    """Tokenize sentence to token, keep only accepted POS tags"""
    words = [tag[0].lower() for tag in nltk.pos_tag(raw_word_tokenizer(text)) if
             len(tag[1]) > 1 and tag[1][:2] in accepted_tags]
    return [word for word in words if not word in STOPWORDS]


def pos_stem_tokenizer(text: Sentence, accepted_tags=["NN", "JJ", "VB"]) -> List[str]:
    return [POTER_STEMMER.stem(word) for word in WORD_TOKENIZER(text) if
            nltk.pos_tag([word])[0][1][:2] in accepted_tags]


def word_tokenizer(text: str) -> List[Word]:
    """Return words from sentence.
    Removing stopwords and punctuation. do lowarcase
    e.g: "Hello World!" >> "hello", "world"
    """

    return WORD_TOKENIZER(text)


def stem_tokenizer(text: str) -> List[Word]:
    """Tokenize and steeming"""
    return [POTER_STEMMER.stem(word) for word in WORD_TOKENIZER(text)]


def stem(word: str):
    return POTER_STEMMER.stem(word)


def noun_phrases(docs: List[str]) -> List[str]:
    _noun_phrases = [chunk.text.lower() for doc in docs for chunk in NLP(doc).noun_chunks]
    return filter_terms(_noun_phrases)


def filter_terms(terms: List[str], remove_stop_words=True, min_count=0, unique=False):
    if remove_stop_words:
        terms = [term for term in terms if term not in STOPWORDS]
    if min_count:
        terms_counts = dict(Counter(terms))
        terms = [term for term in terms if terms_counts[term] >= min_count]
    if unique:
        terms = list(set(terms))

    return terms


class TFIDF:
    """Calculatring tf-idf value

    :param cluster: Target cluster to calculate tf-idf. [token, token]
    :param docs: [[token, token], [token, token], [token, token]]
    """
    token: Term = None

    cluster: List[Term] = []

    documents = List[List[Term]]

    min_count = 1

    def __init__(self, token: Term, cluster: List[Term] = []):
        self.token = token
        if cluster:
            self.cluster = cluster

    @property
    def tfidf(self):
        return self.tf() * self.idf()

    @property
    def tf(self):
        return self._tf(self.token, self.cluster, self.min_count)

    @staticmethod
    def _tf(token: Term, cluster: List[Term], min_count=1):
        """Calculate tf value for a given token"""
        count = cluster.count(token)
        if count < min_count:
            return 0
        return count / len(cluster)

    def __n_docs_contains_token(self):
        return sum(1 for doc in self.documents if self.token in doc)

    @property
    def idf(self):
        return math.log(len(self.documents) / self.__n_docs_contains_token())


class ClusterAnalysis:
    """Analysis of single cluster. Each member of the cluster is n dimentional ndarray
    Find mean_squared_error

    """
    center: np.ndarray = None
    distances: List[float]

    def __init__(self, vectors: List[np.ndarray], distance_metric="cosine_distances"):
        self.vectors = vectors
        self.distance_metric = distance_metric
        self.__set_center()
        self.__calculate_all_distances()

    def __set_center(self):
        self.center = np.mean(self.vectors, axis=0)

    def __distance(self, vector: np.ndarray) -> float:
        """Similarity from center vector"""
        if self.distance_metric == "cosine_distances":
            return cosine_distances([self.center], [vector])[0][0]
        elif self.distance_metric == "euclidean_distances":
            return euclidean_distances([self.center], [vector])[0][0]

    def __calculate_all_distances(self):
        self.distances = [self.__distance(vec) for vec in self.vectors]

    @property
    def sorted_distances(self) -> List[float]:
        """Get sorted distances"""
        return sorted(enumerate(self.distances), key=itemgetter(1))

    @property
    def mean_squared_error(self):
        return sum(distance * distance for distance in self.distances) / len(self.distances)

    @property
    def mse(self):
        """mean_squared_error"""
        return sum(distance * distance for distance in self.distances) / len(self.distances)

    @property
    def sse(self):
        """Sum-of-Squared-Error"""
        return sum(distance * distance for distance in self.distances)

    def boxplot(self):
        from matplotlib import pyplot as plt
        plt.boxplot(self.distances)
        plt.show()
