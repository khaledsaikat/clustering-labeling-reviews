#!/usr/bin/env python3
from nltk.tokenize import sent_tokenize, word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.util import ngrams
from collections import Counter
import nltk
import string

import helpers
import data_loader as dl

class Aspects:

    wordTokens = []

    def __init__(self):
        pass


    def tf(self, n=1):
        '''Calculate term frequency'''
        count = Counter(list(ngrams(self.wordTokens, n))) if n > 1 else Counter(self.wordTokens)
        print(dict(count.most_common(200)))
        self.aspectsOnly(dict(count.most_common(200)))


    def setReviews(self, reviews=[]):
        '''Set revires for single products as list'''
        #self.reviews = helpers.getSingleProductReviews("0972683275")
        self.reviews = helpers.getAllReviewsTexts(reviews)


    def wordTokenize(self):
        '''Tokenize words'''
        tokenizer = RegexpTokenizer(r'\w+')
        [self.wordTokens.extend(tokenizer.tokenize(review.lower())) for review in self.reviews]

        # Tokenize using word_tokenize
        #[self.wordTokens.extend(word_tokenize(review.lower())) for review in self.reviews]

        # Count words and store as dict
        #self.wordsCount = dict(Counter(self.wordTokens))


    def filterStopWords(self):
        self.wordTokens = [word for word in self.wordTokens if not word in stopwords.words('english')]
        # Apply filter to dict
        #self.wordsCount = {word: count for (word, count) in self.wordsCount.items() if not word in stopwords.words('english')}


    def filterPunctuation(self):
        self.wordTokens = [word for word in self.wordTokens if not word in [p for p in string.punctuation]]
        # Apply filter to dict
        #self.wordsCount = {word: count for (word, count) in self.wordsCount.items() if not word in [p for p in string.punctuation]}


    def applyLemmatizing(self):
        lemmatizer = WordNetLemmatizer()
        self.wordTokens = [lemmatizer.lemmatize(word) for word in self.wordTokens]
        # Apply to dict
        #maps = {word: lemmatizer.lemmatize(word) for (word, count) in self.wordsCount.items()}
        #self.combindSimilar(maps)


    def aspectsOnly(self, tokens):
        tokens = {token:count for token, count in tokens.items() if self.isAspect(token)}
        print(tokens)
        print(len(tokens))


    def isAspect(self, token):
        pos = nltk.pos_tag([token])
        if pos[0][1] in ["NN", "NNS"]:
            return True
        else:
            print(pos)


    def combindSimilar(self, maps):
        '''Add values for similar keys'''
        reduced = {}
        for name, count in self.wordsCount.items():
            maped = maps[name]
            if not maped in reduced:
                reduced[maped] = count
            else:
                reduced[maped] = reduced[maped] + count
        self.wordsCount = reduced


    def run(self):
        #self.setReviews()
        self.wordTokenize()
        self.filterStopWords()
        #self.filterPunctuation()
        self.applyLemmatizing()
        self.tf()


if __name__ == "__main__":
    a = Aspects()
    a.setReviews(dl.loadJsonByAspectBraces("data/kindle50_1.json"))
    a.run()
