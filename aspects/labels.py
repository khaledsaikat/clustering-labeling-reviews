#!/usr/bin/env python3
import math
import itertools
from collections import Counter
from pprint import pprint

#from nltk.tokenize import sent_tokenize, word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
import gensim
#from nltk.corpus import wordnet
#from nltk.stem import WordNetLemmatizer
#from nltk.util import ngrams
#import nltk
#import string

from textblob import TextBlob, WordList


#from . import data_loader as dl
import data_loader as dl


class TFIDF:
    '''Calculatring tf-idf value

    :param cluster: Target cluster to calculate tf-idf. [token, token]
    :param clusters: list of cluster which contains list of docs. [[token, token], [token, token]]
    :param docs: [[token, token], [token, token], [token, token]]
    '''
    token = None

    cluster = []

    clusters = []

    docs = []

    selfExcludingClusters = []

    def __init__(self, token, cluster=[], clusters=[]):
        self.token = token
        if cluster:
            self.cluster = cluster
        if clusters:
            self.clusters = clusters


    def tfidf(self):
        return self.tf() * self.idf()


    def tf(self):
        return self._tf(self.token, self.cluster)


    def _tf(self, token, cluster):
        '''Calculate tf value for a given token'''
        return WordList(cluster).count(token) / len(cluster)


    def idf(self):
        #return self.tfBasedIDF()
        #return self.cluserBasedIDF()
        return self.docBasedIDF()


    def _nClustersContainsToken(self):
        return sum(1 for cluster in self.clusters if self.token in cluster)


    def cluserBasedIDF(self):
        return math.log(len(self.clusters) / self._nClustersContainsToken())


    def _nDocsContainsToken(self):
        return sum(1 for doc in self.docs if self.token in doc)


    def docBasedIDF(self):
        return math.log(len(self.docs) / self._nClustersContainsToken())


    def _sumTFExcludingSelfCluster(self):
        clusters = self.selfExcludingClusters
        #clusters = self.clusters.copy()
        #clusters.remove(self.cluster)
        return sum(self._tf(self.token, cluster) for cluster in clusters)


    def tfBasedIDF(self):
        return math.log(len(self.clusters) / (1 + self._sumTFExcludingSelfCluster()))


class Labels:

    clusters = []

    tokenizedClusters = []

    tokensWeight = []

    docs = []

    def __init__(self, clusters):
        '''Contains clusters as a list. Each cluster contains list of lines'''
        self.clusters = clusters

        #print(self.clusters)

        self.normalize()
        self.tokenize()
        self.assignTokensWeight()
        #self.showTopTokens()
        #self.writeTokensWeight()
        #print(self.clusters)


    def normalize(self):
        '''Normalize each doc in cluster'''
        self.clusters = [[self._normalize(doc) for doc in cluster] for cluster in self.clusters]


    def _normalize(self, doc):
        '''Lower text, and filter stop words'''
        doc = TextBlob(doc)
        doc = doc.lower()
        #doc = doc.correct()
        doc = self._filterStopWords(doc)
        return doc


    def _filterStopWords(self, doc):
        words = [word for word in doc.words if not word in stopwords.words('english')]
        return TextBlob(" ".join(words))


    def tokenize(self):
        '''Tokenize words'''
        for index, cluster in enumerate(self.clusters):
            tokens = []
            [tokens.extend(doc.words) for doc in cluster]
            [self.docs.append(list(doc.words)) for doc in cluster]
            self.tokenizedClusters.append(tokens)


    def assignTokensWeight(self):
        TFIDF.clusters = self.tokenizedClusters
        TFIDF.docs = self.docs
        #print(self.clusters)
        #print(TFIDF.docs)
        for cluster in self.tokenizedClusters:
            TFIDF.cluster = cluster
            selfExcludingClusters = self.tokenizedClusters.copy()
            selfExcludingClusters.remove(cluster)
            TFIDF.selfExcludingClusters = selfExcludingClusters
            #self.tokensWeight.append({token: tfidf(token, cluster, self.clusters) for token in set(cluster)})
            self.tokensWeight.append({token: TFIDF(token).tfidf() for token in set(cluster)})


    def getTokensWeight(self):
        '''Get tokens weight ordered by their decending values'''
        return [Counter(tokens) for tokens in self.tokensWeight]


    def writeTokensWeight(self, path="../data/tokens.json"):
        '''Writing tokens weight to file'''
        dl.writeJsonToFile([dict(Counter(tokens)) for tokens in self.tokensWeight], path)


    def showTopTokens(self, aspectsNames=[]):
        [print(aspectsNames[index], Counter(tokens).most_common()) for index, tokens in enumerate(self.tokensWeight)]


    def writeTokens(self, aspectsNames=[]):
        dl.writeJsonToFile({aspectsNames[index]: tokens for index, tokens in enumerate(self.tokensWeight)}, "data/headphone_tokens.json")



def loadSimilarity():
    data = dl.loadJsonFromFile("data/headphone_tokens.json")
    s = Similarity({token: weight for k, v in data.items() for token, weight in v.items()})
    #s = Similarity(data["design"])


if __name__ == "__main__":
    data = {
        "a": ["I'm a  sound bit of a headphone/earphone snob.", "these are super cheap and mostly you get what you pay for."],
        "b": ["Unbelievable sound for the price", "they sound great and are built well."]
    }

    data = dl.loadJsonFromFile("data/headphone_clusters.json")
    clusters = [v for k, v in data.items()]
    cl = Labels(clusters)
    cl.showTopTokens([k for k, v in data.items()])
    #cl.writeTokens([k for k, v in data.items()])
