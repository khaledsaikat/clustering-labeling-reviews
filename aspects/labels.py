#!/usr/bin/env python3
import math
import itertools
from collections import Counter
from pprint import pprint
import operator

from nltk.corpus import stopwords
import gensim
from textblob import TextBlob, WordList

import data_loader as dl
import similar
'''
Find labels for aspects
For interactive loading load python3 from aspects directory
'''


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

    minCount = 1

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
        count = cluster.count(token)
        if count < self.minCount:
            return 0
        return count / len(cluster)
        return WordList(cluster).count(token) / len(cluster)


    def idf(self):
        #return self.tfBasedIDF()
        return self.cluserBasedIDF()
        #return self.docBasedIDF()


    def _nClustersContainsToken(self):
        return sum(1 for cluster in self.clusters if self.token in cluster)


    def cluserBasedIDF(self):
        return math.log(len(self.clusters) / self._nClustersContainsToken())


    def _nDocsContainsToken(self):
        return sum(1 for doc in self.docs if self.token in doc)


    def docBasedIDF(self):
        return math.log(len(self.docs) / self._nDocsContainsToken())


    def _sumTFExcludingSelfCluster(self):
        clusters = self.selfExcludingClusters
        #clusters = self.clusters.copy()
        #clusters.remove(self.cluster)
        return sum(self._tf(self.token, cluster) for cluster in clusters)


    def tfBasedIDF(self):
        return math.log(len(self.clusters) / (1 + self._sumTFExcludingSelfCluster()))


class SingleCluster:
    '''Normalize and tokenize single cluster

    :param cluster: List of normalized docs
    '''
    cluster = []

    def __init__(self, cluster):
        self.cluster = cluster
        self.normalize()
        self.combineTokens()
        print(self.cluster)
        print(self.tokenize())
        print(self.combineTokens())


    def normalize(self):
        '''Normalize each doc in cluster'''
        self.cluster = [self._normalize(doc) for doc in self.cluster]


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


    def tokenize(self, nGram=1):
        '''Tokenize words by docs'''
        return [[tuple(x) for x in doc.ngrams(n=nGram)] for doc in self.cluster]


    def combineTokens(self,  nGram=1):
        '''Combine all tokens in a single list'''
        return [token for doc in self.tokenize(nGram) for token in doc]


class ClusterLabel:

    clusters = []

    def __init__(self, clusters):
        self.clusters = clusters



class NormalizeClusters:
    '''Normalize all clusters

    ???
    :param rawClusters: Raw clusters [[sentence,sentence], [sentence,sentence]]
    :param clusters: Normalized clusters [for cluster in clusters for sentence in cluster] [[sentence,sentence], [sentence,sentence]]
    :param tokenizedClusters: [for cluster in clusters for token in cluster] [[token,token], [token,token]]
    :param tokensWeight: [for cluster in clusters for token, weight in cluster.items()]
    :param docs: Tokenized sentence collections [for sentence in all_sentences for token in sentence]
    '''
    clusters = []

    def __init__(self, clusters):
        '''Contains clusters as a list. Each cluster contains list of docs'''
        self.clusters = clusters
        self.normalize()


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
        words = [word for word in doc.words if not word in stopwords.words("english")]
        return TextBlob(" ".join(words))



class Labels:
    '''Determine label for each clusters

    :param clusters: [for cluster in clusters for sentence in cluster] [[sentence,sentence], [sentence,sentence]]
    :param tokenizedClusters: [for cluster in clusters for token in cluster] [[token,token], [token,token]]
    :param tokensWeight: [for cluster in clusters for token, weight in cluster.items()]
    :param docs: [for sentence in all_sentences for token in sentence]
    '''
    rawClusters = []

    clusters = []

    tokenizedClusters = []

    tokensWeight = []

    docs = []

    def __init__(self, clusters):
        '''Contains clusters as a list. Each cluster contains list of lines'''
        self.rawClusters = clusters
        self.clusters = clusters
        self.normalize()
        self.tokenize()
        self.assignTokensWeight()
        #self.printProperties()
        #self.showTopTokens()
        #self.writeTokensWeight()

    def printProperties(self):
        print("rawClusters:", self.rawClusters)
        print("clusters:", self.clusters)
        print("tokenizedClusters", self.tokenizedClusters)
        print("tokensWeight", self.tokensWeight)
        print("docs", self.docs)

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


    def tokenize(self, nGram=1):
        '''Tokenize words'''
        for index, cluster in enumerate(self.clusters):
            tokens = []
            if nGram>1:
                [tokens.extend([tuple(x) for x in doc.ngrams(n=nGram)]) for doc in cluster]
                [self.docs.append([tuple(x) for x in doc.ngrams(n=nGram)]) for doc in cluster]
            else:
                [tokens.extend(doc.words) for doc in cluster]
                [self.docs.append(list(doc.words)) for doc in cluster]
            self.tokenizedClusters.append(tokens)
        self.docs = [list(x) for x in set(tuple(x) for x in self.docs)]


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
            #self.tokensWeight.append([[token, TFIDF(token).tfidf()] for token in cluster])
            #for token in cluster:
            #    print(token)
            #    print(TFIDF(token).tfidf())
        #print(self.tokensWeight)


    def getTokensWeight(self):
        '''Get tokens weight ordered by their decending values'''
        return [Counter(tokens) for tokens in self.tokensWeight]


    def writeTokensWeight(self, path="../data/tokens.json"):
        '''Writing tokens weight to file'''
        dl.writeJsonToFile([dict(Counter(tokens)) for tokens in self.tokensWeight], path)


    def showTopTokens(self, aspectsNames=[]):
        [print(aspectsNames[index], Counter(tokens).most_common(5)) for index, tokens in enumerate(self.tokensWeight)]


    def writeTokens(self, path="data/tokens.json", aspectsNames=[]):
        '''Writing tokens weight to file'''
        dl.writeJsonToFile({aspectsNames[index]: tokens for index, tokens in Counter(self.tokensWeight).most_common()}, path)
        #dl.writeJsonToFile({aspectsNames[index]: tokens for index, tokens in enumerate(self.tokensWeight)}, path)


    def getSimilarTokens(self, i=0):
        #similar.Similarity.topN = 10
        s = similar.Similarity(self.tokensWeight[i], Counter(self.tokensWeight[i]).most_common(10))
        #print(s.getSimilarTokens())
        #print(Counter(self.tokensWeight[i]).most_common(5))
        tokens = self.recalculateTIFID(s.getReplacableTokens(), i)
        print(Counter(tokens).most_common(5))


    def getSimilarObject(self, i):
        return similar.Similarity(self.tokensWeight[i], Counter(self.tokensWeight[i]).most_common(10))


    def getLabelSentance(self):
        sentWeightClusters = [[self._sentWeight(sent, clusterIndex) for sent in cluster] for clusterIndex, cluster in enumerate(self.clusters)]
        for clusterIndex, sentWeightList in enumerate(sentWeightClusters):
            sentIndex, value = self.maxListValueIndex(sentWeightList)
            print(clusterIndex, self.rawClusters[clusterIndex][sentIndex])


    def maxListValueIndex(self, myList):
        '''Return max value of a list and its index
        :return index, value
        '''
        return max(enumerate(myList), key=operator.itemgetter(1))


    def _sentWeight(self, sent, clusterIndex):
        if len(sent.words) is 0:
            return 0
        return sum(self.tokensWeight[clusterIndex][word] for word in sent.words) / len(sent.words)



    def recalculateTIFID(self, replacableTokens, i):
        TFIDF.clusters = self.replaceSimilarTokensInClusters(replacableTokens)#self.tokenizedClusters
        TFIDF.docs = self.replaceSimilarTokensInDocs(replacableTokens)#self.docs
        TFIDF.cluster = TFIDF.clusters[i]

        selfExcludingClusters = TFIDF.clusters.copy()
        selfExcludingClusters.remove(TFIDF.cluster)
        TFIDF.selfExcludingClusters = selfExcludingClusters

        tokens = {token: TFIDF(token).tfidf() for token in set(TFIDF.cluster)}
        tokens = self.filterAdjAdv(tokens)
        return tokens


    def replaceSimilarTokensInClusters(self, replacableTokens):
        return [[replacableTokens[token] if token in replacableTokens.keys() else token for token in cluster] for cluster in self.tokenizedClusters]


    def replaceSimilarTokensInDocs(self, replacableTokens):
        return [[replacableTokens[token] if token in replacableTokens.keys() else token for token in tokens] for tokens in self.docs]


    def filterAdjAdv(self, tokens):
        return tokens
        return {token: weight for token, weight in tokens.items() if TextBlob(token).tags[0][1] not in ['JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS']}


def getLabelsObject():
    data = dl.loadJsonFromFile("../data/headphone_clusters.json")
    clusters = [v for k, v in data.items()]
    cl = Labels(clusters)
    return cl


def inspectLabels():
    data = {
        "a": ["I'm a  sound bit, of a headphone/earphone snob.", "these are super cheap. and mostly you get what you pay for."],
        "b": ["Unbelievable sound for the price", "they sound great and are built well."]
    }

    data = dl.loadJsonFromFile("../data/headphone_clusters.json")
    clusters = [v for k, v in data.items()]
    cl = Labels(clusters)
    #cl.showTopTokens([k for k, v in data.items()])
    #cl.getSimilarTokens()
    #cl.writeTokens("data/tokens.json", [k for k, v in data.items()])

    #i = 11
    #cl.getSimilarTokens(i)
    #print('Aspect Name:', [k for k, v in data.items()][i])
    for i in range(len(clusters)):
        print(i, [k for k, v in data.items()][i])
        #cl.getSimilarTokens(i)
    cl.getLabelSentance()



if __name__ == "__main__":
    inspectLabels()
