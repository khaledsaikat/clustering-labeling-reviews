#!/usr/bin/env python3
import itertools
from collections import Counter
from pprint import pprint

import gensim
from textblob import TextBlob, WordList

import data_loader as dl
'''
For interactive loading load python3 from aspects directory
To Use Word2Vec:
python3
import similar
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

    topTokens = {}

    similarTokens = []

    def __init__(self, tokens, topTokens):
        self.tokens = self.filterNonWords(tokens)
        self.setTopTokens(topTokens)
        self.similarTokens = []


    def setTopTokens(self, topTokens):
        '''Set top N tokens with removing adjective and adverb

        Don't apply filterAdjAdv to all tokens because it leads to remove important tokens whose are good for sililarity counting
        '''
        self.topTokens = self.filterAdjAdv(self.filterNonWords(dict(topTokens)))


    def filterNonWords(self, tokens):
        '''Filter tokens which are not available on word2vec model'''
        return {token: weight for token, weight in tokens.items() if getWord2VecSimilarity("word", token) is not False}


    def filterAdjAdv(self, tokens):
        return tokens
        return {token: weight for token, weight in tokens.items() if TextBlob(token).tags[0][1] not in ['JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS']}


    def getTokenCombination(self):
        '''Get combination of tokens for top n tokens'''
        return (v for v in itertools.combinations(self.tokens.keys(), 2) if v[0] in self.topTokens or v[1] in self.topTokens)


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


    def getSimilarTokens(self):
        similarity = list(self.getTokensSimilarity())
        self.groupSimilarTokens(similarity)
        #print(self.similarTokens)
        #pprint(self.tokens)
        self.sortSimilarTokens()
        return self.similarTokens


    def getReplacableTokens(self):
        '''Return dict {token_to_replace: replaced_token}'''
        self.getSimilarTokens()
        return {token:tokens[0] for tokens in self.similarTokens for token in tokens}


def loadSimilarity():
    data = dl.loadJsonFromFile("../data/tokens.json")
    #s = Similarity({token: weight for k, v in data.items() for token, weight in v.items()})
    cluster = data["color"]
    print(Counter(cluster).most_common(10))
    s = Similarity(cluster, Counter(cluster).most_common(10))
    pprint(s.getSimilarTokens())
    pprint(s.getReplacableTokens())
    print(s.topTokens)


if __name__ == "__main__":
    loadSimilarity()
