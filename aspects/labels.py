#!/usr/bin/env python3
import math
import itertools
from collections import Counter

#from nltk.tokenize import sent_tokenize, word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
import gensim
#from nltk.corpus import wordnet
#from nltk.stem import WordNetLemmatizer
#from nltk.util import ngrams
#import nltk
#import string

from textblob import TextBlob, WordList


#import helpers
import data_loader as dl


WORD2VEC_MODEL = None

def tf(token, cluster):
    '''Calculate tf value for a given token

    :param token: input token as string
    :param cluster: list of token
    :returns: tf value
    '''
    return WordList(cluster).count(token) / len(cluster)


def _n_containing(token, clusters):
    return sum(1 for cluster in clusters if token in cluster)


def idf(token, clusters):
    return math.log(len(clusters) / (0 + _n_containing(token, clusters)))


def customIDF(token, cluster, clusters):
    return math.log(len(clusters) / (0 + _combindTF(token, cluster, clusters)))


def _combindTF(token, cluster, clusters):
    #if cluster in clusters: clusters.remove(cluster)
    return sum(_tf(token, c) for c in clusters)


def tfidf(token, cluster, clusters):
    #return  self._tf(token, cluster) * self.customIDF(token, cluster, self.clusters)
    return  tf(token, cluster) * idf(token, clusters)


def getTokenCombination(tokens=[], firstN=20):
    '''Get combination of tokens for firstN values'''
    return (v for v in itertools.combinations(tokens, 2) if v[0] in tokens[:firstN])


def getSimilarTokens(tokens, minSimilarity=0.3):
    combinations = getTokenCombination(tokens, 10)
    for v in combinations:
        similarity = word2VecModel.similarity(v[0], v[1])
        if similarity >= minSimilarity:
            yield (v[0], v[1], similarity)


def reduceSimilarTokens(tokensWeight):
    similarTokens = list(getSimilarTokens(list(tokensWeight.keys())))
    similarTokens.reverse()
    for t in similarTokens:
        removedWeight = tokensWeight[t[1]]
        tokensWeight[t[0]] += removedWeight
        del tokensWeight[t[1]]
    return tokensWeight



class Labels:

    clusters = []

    tokensWeight = []

    def __init__(self, clusters):
        '''Contains clusters as a list. Each cluster contains list of lines'''
        self.clusters = clusters

        #print(self.clusters)

        self.normalize()
        self.tokenize()
        self.assignTokensWeight()
        self.showTopTokens()
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
            self.clusters[index] = tokens


    def assignTokensWeight(self):
        for cluster in self.clusters:
            self.tokensWeight.append({token: tfidf(token, cluster, self.clusters) for token in set(cluster)})


    def showTopTokens(self):
        [print(Counter(tokens).most_common(20)) for tokens in self.tokensWeight]
        dl.writeJsonToFile([dict(Counter(tokens).most_common(20)) for tokens in self.tokensWeight], "../data/temp.json")



if __name__ == "__main__":
    #model = gensim.models.KeyedVectors.load_word2vec_format("/Users/khaled/nltk_data/GoogleNews-vectors-negative300.bin", binary=True)
    samples = [
        ["I'm a  sound bit of a headphone/earphone snob.", "these are super cheap and mostly you get what you pay for."],
        ["Unbelievable sound for the price", "they sound great and are built well."]
    ]

    data = dl.loadJsonFromFile("../data/headphone_clusters.json")
    cluster = [v for v in data.values()]
    cl = Labels(cluster)
    #cl = ClusterLabel(samples)
    #a.setReviews(dl.loadJsonByAspectBraces("data/kindle501.json"))
    #a.run()
