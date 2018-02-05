import math
from typing import Tuple, Union, List

import nltk
from gensim.parsing.porter import PorterStemmer
from gensim.parsing.preprocessing import STOPWORDS
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from textblob import TextBlob

Word = str
Sentence = str
Term = Union[Word, Tuple[Word, Word], Tuple[Word, Word, Word]]
Document = TextBlob

WORD_TOKENIZER = CountVectorizer(stop_words="english").build_analyzer()

REGEXP_TOKENIZER = RegexpTokenizer("\w+|\$[\d\.]+|\S+")

POTER_STEMMER = PorterStemmer()


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
    return [POTER_STEMMER.stem(word) for word in WORD_TOKENIZER(text) if nltk.pos_tag([word])[0][1][:2] in accepted_tags]


def word_tokenizer(text: str) -> List[Word]:
    """Return words from sentence.
    Removing stopwords and punctuation. do lowarcase
    e.g: "Hello World!" >> "hello", "world"
    """

    return WORD_TOKENIZER(text)


def stem_tokenizer(text: str) -> List[Word]:
    """Tokenize and steeming"""
    return [POTER_STEMMER.stem(word) for word in WORD_TOKENIZER(text)]


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
