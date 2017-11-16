#!/usr/bin/env python3
import operator
from typing import List, Tuple, Union, Dict

import data_loader as dl
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
TokenizeDocument = List[Term]
DocumentsCluster = List[Document]
TokensCluster = List[Term]
TokenizeDocCluster = List[List[Term]]


class NormalizeClusters:
    """Normalize all clusters"""

    def __init__(self, clusters: List[List[str]]):
        self.__clusters = []
        self.normalize(clusters)

    @property
    def clusters(self) -> List[DocumentsCluster]:
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
    __cluster: TokenizeDocCluster = []

    def __init__(self, cluster: DocumentsCluster, n_gram: int = 1):
        self.n_gram: int = n_gram
        self.__tokenize(cluster)

    def __tokenize(self, cluster: DocumentsCluster):
        if self.n_gram > 1:
            self.__cluster = [[tuple(x) for x in doc.ngrams(n=self.n_gram)] for doc in cluster]
        else:
            self.__cluster = [doc.words for doc in cluster]

    @property
    def cluster(self) -> TokenizeDocCluster:
        """Return tokenized documents"""
        return self.__cluster

    @property
    def tokens(self) -> TokensCluster:
        """Combined all tokens"""
        return [term for doc in self.__cluster for term in doc]


class TokenizeClusters:
    """Tokenize all clusters"""

    def __init__(self, clusters: List[DocumentsCluster], n_gram: int = 1):
        self.__clusters: List[TokenizeSingleCluster] = []
        self.n_gram: int = n_gram
        self.__tokenize(clusters)

    def __tokenize(self, clusters: List[DocumentsCluster]):
        self.__clusters = [TokenizeSingleCluster(cluster, self.n_gram) for cluster in clusters]

    @property
    def clusters(self) -> List[TokenizeSingleCluster]:
        """Get list of tokenized cluster object"""
        return self.__clusters

    @property
    def documents_clusters(self) -> List[TokenizeDocCluster]:
        """Get clusters of tokenized documents"""
        return [[doc for doc in cluster_object.cluster] for cluster_object in self.__clusters]

    @property
    def tokens_clusters(self) -> List[TokensCluster]:
        """Get clusters of tokens"""
        return [[token for token in cluster_object.tokens] for cluster_object in self.__clusters]

    @property
    def documents(self) -> List[TokenizeDocument]:
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


class ClustersTokensWeight:
    """Get clusters of tokens weight"""

    def __init__(self, clusters: List[DocumentsCluster], n_gram: int):
        self.tokenize_clusters = TokenizeClusters(clusters, n_gram)
        self.idf_weights = idf_tokens_weights(self.tokenize_clusters)
        self.clusters_tokens_weights = self.calculate_tokens_weights()

    def calculate_tokens_weights(self) -> List[Dict[Term, float]]:
        return [{token: self.__calculate_tfidf(token, cluster) for token in cluster} for cluster
                in self.tokenize_clusters.tokens_clusters]

    def __calculate_tfidf(self, token: Term, cluster: TokensCluster) -> float:
        tf = TFIDF(token, cluster).tf
        return tf * self.idf_weights[token]

    @property
    def clusters(self) -> List[Dict[Term, float]]:
        return self.clusters_tokens_weights


class UnigramSentenceWeightModel:
    """Clusters of unigram weighted sentences"""

    def __init__(self, clusters: List[DocumentsCluster]):
        self.clusters_documents: List[DocumentsCluster] = clusters
        self.clusters_tokens_weight: List[Dict[Term, float]] = ClustersTokensWeight(clusters, 1).clusters

    def single_sentence_weight(self, sentence: Document, cluster_index: int) -> float:
        return self.__sum_tokens_weight(sentence.words, cluster_index)

    def __sum_tokens_weight(self, tokens: List[Term], cluster_index: int) -> float:
        if len(tokens) < 1:
            return 0
        return sum(self.clusters_tokens_weight[cluster_index][token] for token in tokens) / len(tokens)

    @property
    def clusters(self) -> List[List[float]]:
        return [[self.single_sentence_weight(doc, cluster_index) for doc in cluster] for cluster_index, cluster in
                enumerate(self.clusters_documents)]


class MultigramSentenceWeightModel:
    """Clusters of multigram weighted sentences"""

    def __init__(self, clusters: List[DocumentsCluster]):
        self.clusters_documents: List[DocumentsCluster] = clusters
        self.clusters_tokens_weight: List[Dict[Term, float]] = self.combine_tokens_weight(clusters)

    @staticmethod
    def combine_tokens_weight(clusters: List[DocumentsCluster]):
        unigram_clusters = ClustersTokensWeight(clusters, 1).clusters
        bigram_clusters = ClustersTokensWeight(clusters, 2).clusters
        trigram_clusters = ClustersTokensWeight(clusters, 3).clusters
        for index, cluster in enumerate(unigram_clusters):
            unigram_clusters[index].update(bigram_clusters[index])
            unigram_clusters[index].update(trigram_clusters[index])

        return unigram_clusters

    def single_sentence_weight(self, sentence: Document, cluster_index: int) -> float:
        uigram = sentence.words
        bigram = [tuple(x) for x in sentence.ngrams(n=2)]
        trigram = [tuple(x) for x in sentence.ngrams(n=3)]

        return self.__sum_tokens_weight(uigram, cluster_index) \
               + self.__sum_tokens_weight(bigram, cluster_index) \
               + self.__sum_tokens_weight(trigram, cluster_index)

    def __sum_tokens_weight(self, tokens: List[Term], cluster_index: int) -> float:
        if len(tokens) < 1:
            return 0
        return sum(self.clusters_tokens_weight[cluster_index][token] for token in tokens) / len(tokens)

    @property
    def clusters(self) -> List[List[float]]:
        return [[self.single_sentence_weight(doc, cluster_index) for doc in cluster] for cluster_index, cluster in
                enumerate(self.clusters_documents)]


class ClustersLabels:
    def __init__(self, raw_clusters: List[List[str]]):
        self.raw_clusters: List[List[str]] = raw_clusters
        self.clusters_documents: List[DocumentsCluster] = NormalizeClusters(raw_clusters).clusters
        #self.clusters_weights: List[List[float]] = UnigramSentenceWeightModel(self.clusters_documents).clusters
        self.clusters_weights: List[List[float]] = MultigramSentenceWeightModel(self.clusters_documents).clusters

    @staticmethod
    def __max_value(my_list) -> Tuple[int, float]:
        """Return max value of a list and its index"""
        return max(enumerate(my_list), key=operator.itemgetter(1))

    @property
    def clusters_index(self) -> List[Tuple[int, float]]:
        return [self.__max_value(cluster) for cluster in self.clusters_weights]

    @property
    def clusters(self) -> List[Document]:
        return [self.raw_clusters[cluster_index][self.__max_value(cluster)[0]] for cluster_index, cluster in
                enumerate(self.clusters_weights)]


class W2VSingleCluster:
    cluster: DocumentsCluster = []

    # 300 dimension numpy.ndarray
    docs_vector = []

    center = []

    def __init__(self, cluster: DocumentsCluster):
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

    def get_closest_doc_index(self) -> int:
        weights = [self.distance(self.center, vec) for vec in self.docs_vector]
        return min(enumerate(weights), key=operator.itemgetter(1))[0]

    @property
    def doc_index(self) -> int:
        return self.get_closest_doc_index()


class W2VClustersLabels:
    def __init__(self, raw_clusters: List[List[str]]):
        self.raw_clusters: List[List[str]] = raw_clusters
        self.clusters_documents: List[DocumentsCluster] = NormalizeClusters(raw_clusters).clusters

    @property
    def clusters(self) -> List[Document]:
        return [self.raw_clusters[cluster_index][W2VSingleCluster(cluster).doc_index] for cluster_index, cluster in
                enumerate(self.clusters_documents)]


def compare_labels(pre_labels: List[str], post_labels: List[str]):
    for index, label in enumerate(pre_labels):
        print(label, ":", post_labels[index])


def inspect_labels():
    data = {
        "a": ["I'm a  sound bit, of a headphone/earphone snob.",
              "these are super cheap. and mostly you get what you pay for."],
        "b": ["Unbelievable sound for the price", "they sound great and are built well."]
    }
    data = dl.loadJsonFromFile("../data/headphone_clusters.json")

    pre_labels = [k for k, v in data.items()]
    raw_clusters = [v for k, v in data.items()]

    labels = ClustersLabels(raw_clusters).clusters
    # labels = W2VClustersLabels(raw_clusters).clusters
    compare_labels(pre_labels, labels)


if __name__ == "__main__":
    # w2v.load_word2vec_model()
    inspect_labels()
