#!/usr/bin/env python3
import json

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

    return [review["reviewText"] for review in reviews]
