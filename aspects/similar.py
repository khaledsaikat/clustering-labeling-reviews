#!/usr/bin/env python3
import itertools
from collections import Counter
from pprint import pprint

import gensim
from textblob import TextBlob, WordList

from . import data_loader as dl
'''
To Use Word2Vec:
python3
from aspects import similar
similar.loadWord2VecModel()
model = similar.WORD2VEC_MODEL
similar.loadSimilarity()

# Reloading
import imp
def reload():
    imp.reload(similar)
    similar.WORD2VEC_MODEL = model
    similar.loadSimilarity()
reload()

'''

WORD2VEC_MODEL = None

SOURCE_PATH = "/Users/khaled/nltk_data/GoogleNews-vectors-negative300.bin"

def loadWord2VecModel():
    '''Loading GoogleNews-vectors-negative300 traind model'''
    global WORD2VEC_MODEL
    WORD2VEC_MODEL = gensim.models.KeyedVectors.load_word2vec_format(SOURCE_PATH, binary=True)


def getWord2VecSimilarity(word1, word2):
    try:
        return WORD2VEC_MODEL.similarity(word1, word2)
    except Exception as e:
        return False


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


    def getSimilarTokens(self):
        similarity = list(self.getTokensSimilarity())
        self.groupSimilarTokens(similarity)
        self.sortSimilarTokens()
        return self.similarTokens


    def setTopNTokens(self):
        '''Set top N tokens with removing adjective and adverb

        Don't apply filterAdjAdv to all tokens because it leads to remove important tokens whose are good for sililarity counting
        '''
        #tokens = Counter(self.filterAdjAdv())
        self.topNTokens = Counter({v[0]:v[1] for v in self.tokens.most_common(self.topN)})


    def filterNonWords(self):
        '''Filter tokens which are not available on word2vec model'''
        self.tokens = Counter({token: weight for token, weight in self.tokens.items() if getWord2VecSimilarity("word", token) is not False})


    def filterAdjAdv(self):
        return {token: weight for token, weight in self.tokens.items() if TextBlob(token).tags[0][1] not in ['JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS']}


    def getTokenCombination(self):
        '''Get combination of tokens for top n tokens'''
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


    def sortSimilarTokens(self):
        '''similarTokens'''
        self.similarTokens = [sorted(tokens, key=lambda x: self.tokens[x], reverse=True) for tokens in self.similarTokens]


def loadSimilarity():
    data = dl.loadJsonFromFile("data/headphone_tokens.json")
    #s = Similarity({token: weight for k, v in data.items() for token, weight in v.items()})
    s = Similarity(data["noise cancellation"])
    pprint(s.getSimilarTokens())


if __name__ == "__main__":
    loadSimilarity()
