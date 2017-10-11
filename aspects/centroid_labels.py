#!/usr/bin/env python3
import operator
from typing import List, Tuple, Union, Dict

import numpy as np
import word2vec as w2v
from nltk.corpus import stopwords
from textblob import TextBlob
from utils import TFIDF

"""Find clusters labels by using centroid

For interactive loading, load python3 from aspects directory
```
import word2vec as w2v
import centroid_labels as cl
w2v.load_word2vec_model()
cl.inspect_labels()
```
"""

Word = str
Term = Union[Word, Tuple[Word, Word], Tuple[Word, Word, Word]]
Document = TextBlob
DocumentsCluster = List[Document]
TokensCluster = List[Term]
TokenizeDocCluster = List[List[Term]]


class NormalizeClusters:
    """Normalize all clusters"""

    def __init__(self, clusters: List[List[str]]):
        self.__clusters = []
        self.normalize(clusters)

    @property
    def clusters(self) -> List[List[Document]]:
        return self.__clusters

    def normalize(self, clusters: List[List[str]]):
        """Normalize each doc in cluster"""
        self.__clusters = [[self.__normalize(doc) for doc in cluster] for cluster in clusters]

    def __normalize(self, doc: str) -> TextBlob:
        """Lower text, and filter stop words"""
        doc = TextBlob(doc)
        doc = doc.lower()
        # doc = doc.correct()
        doc = self.__filter_stop_words(doc)
        return doc

    @staticmethod
    def __filter_stop_words(doc: TextBlob) -> TextBlob:
        words = [word for word in doc.words if not word in stopwords.words("english")]
        return TextBlob(" ".join(words))


class TokenizeSingleCluster:
    __cluster: List[List[Term]] = []

    def __init__(self, cluster: List[Document], n_gram: int = 1):
        self.n_gram: int = n_gram
        self.__tokenize(cluster)

    def __tokenize(self, cluster: List[Document]) -> List[List[Term]]:
        if self.n_gram > 1:
            self.__cluster = [[tuple(x) for x in doc.ngrams(n=self.n_gram)] for doc in cluster]
        else:
            self.__cluster = [doc.words for doc in cluster]

    @property
    def cluster(self) -> List[List[Term]]:
        """Return tokenized documents"""
        return self.__cluster

    @property
    def tokens(self) -> List[Term]:
        """Combined all tokens"""
        return [term for doc in self.__cluster for term in doc]


class TokenizeClusters:
    """Tokenize all clusters"""

    def __init__(self, clusters: List[List[Document]], n_gram: int = 1):
        self.__clusters: List[TokenizeSingleCluster] = []
        self.n_gram: int = n_gram
        self.__tokenize(clusters)

    def __tokenize(self, clusters: List[List[Document]]):
        self.__clusters = [TokenizeSingleCluster(cluster, self.n_gram) for cluster in clusters]

    @property
    def clusters(self) -> List[TokenizeSingleCluster]:
        """Get list of tokenized cluster object"""
        return self.__clusters

    @property
    def documents_clusters(self) -> List[List[List[Term]]]:
        """Get clusters of tokenized documents"""
        return [[doc for doc in cluster_object.cluster] for cluster_object in self.__clusters]

    @property
    def tokens_clusters(self) -> List[List[Term]]:
        """Get clusters of tokens"""
        return [[token for token in cluster_object.tokens] for cluster_object in self.__clusters]

    @property
    def documents(self) -> List[List[Term]]:
        """Get list of all tokenized documents"""
        return [doc for cluster_object in self.__clusters for doc in cluster_object.cluster]

    @property
    def tokens(self) -> List[Term]:
        """Get list of all tokens"""
        return set(token for doc in self.documents for token in doc)


def idf_tokens_weights(clusters: TokenizeClusters) -> Dict[Term, float]:
    """Get dictionary of token: idf_weight"""
    TFIDF.documents = clusters.documents
    return {token: TFIDF(token).idf for token in clusters.tokens}


class UnigramCentroidLabel:
    idf_weights: Dict[Term, float] = {}

    clusters_tokens_weights: List[Dict[Term, float]] = []

    def __init__(self, clusters: List[DocumentsCluster]):
        self.tokenized_clusters = TokenizeClusters(clusters)
        self.idf_weights = idf_tokens_weights(self.tokenized_clusters)
        self.calculate_tokens_weights()

    def calculate_tokens_weights(self):
        self.clusters_tokens_weights = [{token: self.__calculate_tfidf(token, cluster) for token in cluster} for cluster
                                        in self.tokenized_clusters.tokens_clusters]

    def __calculate_tfidf(self, token: Term, cluster: TokensCluster):
        tf = TFIDF(token, cluster).tf
        return tf * self.idf_weights[token]

    def calculate_sentence_weight(self, cluster: TokenizeDocCluster):
        pass


class LabelSentance:
    def __init__(self, cluster):
        pass


def get_label_sentance(self):
    sent_weight_clusters = [[self.__sent_weight(sent, cluster_index) for sent in cluster] for clusterIndex, cluster
                            in
                            enumerate(self.clusters)]
    for cluster_index, sent_weight_list in enumerate(sent_weight_clusters):
        sent_index, value = self.max_list_value_index(sent_weight_list)
        print(cluster_index, self.rawClusters[cluster_index][sent_index])


def max_list_value_index(self, my_list):
    """Return max value of a list and its index"""
    return max(enumerate(my_list), key=operator.itemgetter(1))


def __sent_weight(self, sent, cluster_index):
    if len(sent.words) is 0:
        return 0
    return sum(self.tokensWeight[cluster_index][word] for word in sent.words) / len(sent.words)


class W2VCentroidLabel:
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


class CentroidLabel(W2VCentroidLabel):
    """Act as a wrapper of parent class, so that we can change parent according to need"""
    pass


class ClustersCentroidLabels:
    def __init__(self, clusters: List[List[str]]):
        self.raw_clusters: List[List[str]] = clusters
        self.clusters = NormalizeClusters(clusters).clusters

    def process(self):
        UnigramCentroidLabel(self.clusters)


def inspect_labels():
    data = {
        "a": ["I'm a  sound bit, of a headphone/earphone snob.",
              "these are super cheap. and mostly you get what you pay for."],
        "b": ["Unbelievable sound for the price", "they sound great and are built well."]
    }
    # data = dl.loadJsonFromFile("../data/headphone_clusters.json")

    pre_labels = [k for k, v in data.items()]
    raw_clusters = [v for k, v in data.items()]

    clusters = NormalizeClusters(raw_clusters).clusters
    # clusters = TokenizeClusters(clusters).documents_clusters
    # print(clusters)
    # return

    UnigramCentroidLabel(clusters)
    return

    for index, cluster in enumerate(clusters):
        cl = CentroidLabel(cluster)
        label = raw_clusters[index][cl.get_closest_doc_index()]
        print(pre_labels[index], ":", label)


if __name__ == "__main__":
    # w2v.load_word2vec_model()
    inspect_labels()
