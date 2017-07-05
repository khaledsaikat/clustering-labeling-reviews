#!/usr/bin/env python3
import json
import math
from collections import Counter
from textblob import TextBlob

def loadJsonToList(path):
    """Load json from file. (each line contains single json string).
    Put all json to a list
    """

    data = []
    with open(path) as json_file:
        for line in json_file:
            data.append(json.loads(line))

    return data


def groupByProducts(reviews):
    """Group reviews by products"""
    data = {}
    for review in reviews:
        if not review["asin"] in data:
            data[review["asin"]] = []
        data[review["asin"]].append(review)

    return data


def getSingleProductReviews(productID=None):
    """Get all reviews text by product id(asin)"""
    allReviews = groupByProducts(loadJsonToList("data/reviews.json"))
    if not productID:
        productID = list(allReviews.keys())[0]
    reviews = allReviews[productID]

    #return [review for review in reviews]
    return [review["reviewText"] for review in reviews]


def getAllReviewsTexts(reviews=[]):
    #if not reviews:
    #    reviews = loadJsonToList("data/examples.txt")
    return [review["reviewText"] for review in reviews]


def getManualAspects(reviews=[]):
    aspects = []
    [aspects.extend(v["aspects"]) for v in reviews]
    return Counter(aspects)


def groupByAspect(reviews=[]):
    aspects = {}
    for review in reviews:
        for aspect, line in review["aspects"].items():
            if aspect not in aspects:
                aspects[aspect] = []
            aspects[aspect].append(review)
    return aspects


def getTopAspectGroups(reviews, n=10):
    aspects = groupByAspect(reviews)
    aspectsCounts = {k:len(v) for k,v in aspects.items()}
    aspectsNames = sorted(aspectsCounts, key=aspectsCounts.__getitem__, reverse=True)[:n]
    aspectsList = [aspects[aspect] for aspect in aspectsNames]
    #return aspectsList
    return [TextBlob(" ".join(list(map(lambda x: x["reviewText"], grp)))) for grp in aspectsList]


def tf(word, blob):
    return blob.words.count(word) / len(blob.words)


def n_containing(word, bloblist):
    return sum(1 for blob in bloblist if word in blob.words)


def idf(word, bloblist):
    return math.log(len(bloblist) / (1 + n_containing(word, bloblist)))


def tfidf(word, blob, bloblist):
    return tf(word, blob) * idf(word, bloblist)

#r = getAllReviewsTexts()
#print(r[1])
# r = re.findall(r"\{.*?aspects.*?\}.*?\}", f1, re.DOTALL)
