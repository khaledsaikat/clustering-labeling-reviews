#!/usr/bin/env python3
import operator
from pprint import pprint
from typing import List, Tuple, Callable, Union
from collections import Counter

import numpy as np
import word2vec as w2v
from cluster import AgglomerativeClustering, group_result
#from matplotlib import pyplot as plt
from nltk.tokenize import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from nltk.util import bigrams
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
import re
import gensim
from nltk import RegexpTokenizer
from nltk.corpus import stopwords
from textblob import TextBlob


def run_20ng_sample():
    categories = ["rec.sport.hockey"]
    documents = fetch_20newsgroups(subset="train", categories=[categories[0]]).data[:50]
    documents = [_cleanup_20ng_doc(doc) for doc in documents]
    #documents = [sent_tokenize(doc) for doc in documents]
    #all_sentences = [sent for doc in documents for sent in doc]
    #return all_sentences
    cluster_text = " ".join(doc for doc in set(documents))
    run(cluster_text)


def _cleanup_20ng_doc(document: str):
    document = re.sub("From:.+\n", "", document)
    document = re.sub("Lines:.+\n", "", document)
    document = document.replace("\n", " ")
    return document.lower()


def run_clusters_sample():
    import data_loader as dl
    data = dl.loadJsonFromFile("../data/headphone_clusters.json")
    clusters = data.values()
    terms_clusters = [[TextBlob(sent).noun_phrases for sent in cluster] for cluster in clusters]
    print(terms_clusters[3], len(terms_clusters[3]), len(terms_clusters))
    return

    # all documents / sentences
    docs = list(set(doc for cluster in data.values() for doc in cluster))
    # cluster as whole text
    clusters_text = [" ".join(doc for doc in set(cluster)) for cluster in data.values()]
    all_terms = extract_all_terms(sent_clusters)
    pprint(Counter(all_terms).most_common(10))
    #run(clusters_text[5])


def extract_all_terms(sent_clusters: List[List[str]]):
    terms = []
    for cluster in sent_clusters:
        for sent in cluster:
            terms.append(list(TextBlob(sent).noun_phrases))

    return terms


def run(cluster_text: str):
    cluster_text_blob = TextBlob(cluster_text)
    pprint(Counter(cluster_text_blob.noun_phrases).most_common(10))


#run_clusters_sample()
#run_20ng_sample()



def test():
    cv = CountVectorizer(stop_words="english")
    tokenizer = cv.build_analyzer()
    import data_loader as dl
    data = dl.loadJsonFromFile("../data/headphone_clusters.json")
    clusters = data.values()
    all_sentences = [tokenizer(sent) for cluster in clusters for sent in cluster]
    print(all_sentences[1], len(all_sentences))
    model = gensim.models.Word2Vec(all_sentences, min_count=1)
    print(model.most_similar("quality"))
    return model

test()
