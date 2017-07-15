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


#import helpers
from . import data_loader as dl

'''
To Use Word2Vec:
python3
form aspects import labels
labels.loadWord2VecModel()
model = labels.WORD2VEC_MODEL
labels.loadSimilarity()

# Reloading
from imp import reload
def load():
    reload(labels)
    labels.WORD2VEC_MODEL = model
    labels.loadSimilarity()
load()

'''



WORD2VEC_MODEL = None

TOKEN_MAP = {}


def loadWord2VecModel():
    '''Loading GoogleNews-vectors-negative300 traind model'''
    global WORD2VEC_MODEL
    WORD2VEC_MODEL = gensim.models.KeyedVectors.load_word2vec_format("/Users/khaled/nltk_data/GoogleNews-vectors-negative300.bin", binary=True)


def getWord2VecSimilarity(word1, word2):
    try:
        return WORD2VEC_MODEL.similarity(word1, word2)
    except Exception as e:
        return False


def reduceSimilarTokens(tokensWeight):
    '''Find and reduce similar tokens

    :param tokensWeight: tokens for single cluster
    '''

    similarTokens = list(getSimilarTokens(list(tokensWeight.keys())))
    similarTokens.reverse()
    for t in similarTokens:
        removedWeight = tokensWeight[t[1]]
        tokensWeight[t[0]] += removedWeight
        del tokensWeight[t[1]]
        updateTokenMap(t[0], t[1])
    return tokensWeight


def updateTokenMap(original, synoname):
    global TOKEN_MAP
    if original in TOKEN_MAP:
        TOKEN_MAP[original].append(synoname)
    else:
        TOKEN_MAP.update({original:[synoname]})


class Similarity:
    '''Calculate similarity

    :param baseTokens: Base tokens with tf-idf weight {token1:0.123, token2:0.123}
    '''

    tokens = {}

    topNTokens = {}

    topN = 100

    similarTokens = []


    def __init__(self, tokens):
        self.tokens = tokens
        self.filterNonWords()
        self.setTopNTokens()
        print(self.topNTokens)
        #return
        similarity = list(self.getTokensSimilarity())
        self.groupSimilarTokens(similarity)
        pprint(self.similarTokens)

        self.mergeToTopNTokens()

        #print(len(list(self.getTokenCombination())))
        #print(list(self.getSimilarTokens(0)))


    def setTopNTokens(self):
        '''Set top N tokens with removing adjective and adverb

        Don't apply filterAdjAdv to all tokens because it leads to remove important tokens whose are good for sililarity counting
        '''
        tokens = Counter(self.filterAdjAdv())
        self.topNTokens = Counter({v[0]:v[1] for v in tokens.most_common(self.topN)})


    def filterNonWords(self):
        self.tokens = {token: weight for token, weight in self.tokens.items() if getWord2VecSimilarity("word", token) is not False}


    def filterAdjAdv(self):
        return {token: weight for token, weight in self.tokens.items() if TextBlob(token).tags[0][1] not in ['JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS']}


    def getTokenCombination(self):
        '''Get combination of tokens for firstN values'''
        return (v for v in itertools.combinations(self.tokens.keys(), 2) if v[0] in self.topNTokens or v[1] in self.topNTokens)


    def getTokensSimilarity(self, minSimilarity=0.5):
        '''Return generator of ('cheaper', 'price', 0.36297958784528356)'''
        combinations = self.getTokenCombination()
        for v in combinations:
            similarity = getWord2VecSimilarity(v[0], v[1])
            if similarity >= minSimilarity:
                yield (v[0], v[1], similarity)


    def groupSimilarTokens(self, tokensSimilarity):
        for similarity in tokensSimilarity:
            if self._isTokenAdded(similarity[0]) is False and self._isTokenAdded(similarity[1]) is False:
                self.similarTokens.append([similarity[0], similarity[1]])
            else:
                token1index = self._isTokenAdded(similarity[0])
                token2index = self._isTokenAdded(similarity[1])
                if token1index is not False and token2index is False:
                    self.similarTokens[token1index].append(similarity[1])
                elif token2index is not False and token1index is False:
                    self.similarTokens[token2index].append(similarity[1])
        self.similarTokens = [list(set(tokens)) for tokens in self.similarTokens]


    def _isTokenAdded(self, token):
        for index, tokens in enumerate(self.similarTokens):
            if token in tokens:
                return index
        return False


    def mergeToTopNTokens(self):
        tokensLabel = self.LabelSimilarTokens()
        for token in self.topNTokens.most_common(self.topN):
            if token[0] in tokensLabel:
                self.topNTokens[token[0]] = sum(self.tokens[t] for t in tokensLabel[token[0]])
        print(self.topNTokens)



    def LabelSimilarTokens(self):
        similarTokensLabel = {}
        for token in self.topNTokens.most_common(self.topN):
            index = self._isTokenAdded(token[0])
            if index is not False:
                similarTokensLabel[token[0]] = self.similarTokens[index]
        #pprint(similarTokensLabel)
        return similarTokensLabel



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

    #data = dl.loadJsonFromFile("data/headphone_clusters.json")
    clusters = [v for k, v in data.items()]
    #cl = Labels(clusters)
    #cl.showTopTokens([k for k, v in data.items()])
    #cl.writeTokens([k for k, v in data.items()])

    data = dl.loadJsonFromFile("data/headphone_tokens.json")
    [print(k, Counter(v).most_common(5)) for k,v in data.items()]

    s = Similarity(data["price"])

    #tokens = [('price', 0.09259364473953426), ('sound', 0.06760968694703753), ('quality', 0.05701842669686391)]
    #s = Similarity(tokens)
