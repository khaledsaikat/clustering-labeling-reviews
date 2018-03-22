#!/usr/bin/env python3
# Clusters labeling
#
# Load w2v first!!
##

import math
from collections import Counter
from operator import itemgetter
from typing import List

import data_loader as dl
import labeling_utils
import utils
import word2vec as w2v
from gensim.parsing.preprocessing import STOPWORDS
from models import AgglomerativeClustering, group_result
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.pipeline import make_pipeline


def get_kindle_clusters() -> List[List[str]]:
    """Get sentences as a list"""
    reviews = dl.load_json_from_file("../data/kindle_500.json")
    return [sent[1] for review in reviews for sent in enumerate(review)]


def get_headphone_clusters() -> List[List[str]]:
    data = dl.load_json_from_file("../data/headphone_100_clusters.json")
    return [cluster[0] for cluster in data]


def get_headphone_reviews():
    data = dl.load_json_from_file("../data/headphone100.json")
    return [review["reviewText"] for review in data]


def baseline_labeling():
    """Baseline labeling"""
    reviews = get_headphone_reviews()
    tfidf_vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(2, 2))
    tfidf_vectorizer.fit(reviews)
    features = tfidf_vectorizer.get_feature_names()

    clusters = list(get_headphone_clusters())
    del clusters[-1]
    text_clusters = [" ".join(cluster) for cluster in clusters]
    vectorized_clusters = tfidf_vectorizer.transform(text_clusters)

    clusters_top_terms = []
    for vectorized_cluster in vectorized_clusters.todense():
        weights = sorted(enumerate(vectorized_cluster.tolist()[0]), key=itemgetter(1), reverse=True)
        clusters_top_terms.append([(features[token[0]], token[1]) for token in weights[:10]])

    print(evaluate_clusters([[label[0] for label in labels] for labels in clusters_top_terms]))


def stem_labeling(apply_sub_clustering=False):
    """Baseline labeling"""
    reviews = get_headphone_reviews()
    tfidf_vectorizer = TfidfVectorizer(stop_words=STOPWORDS, tokenizer=utils.stem_tokenizer)
    tfidf_vectorizer.fit(reviews)
    features = tfidf_vectorizer.get_feature_names()

    clusters = list(get_headphone_clusters())
    del clusters[-1]

    if apply_sub_clustering:
        clusters_labels = []
        for cluster in clusters:
            sub_clusters = sub_clustering(cluster)
            sub_clusters_size_ratio = [(len(cl) / len(cluster)) for cl in sub_clusters]
            text_sub_clusters = [" ".join(cl) for cl in sub_clusters]
            vectorized_sub_clusters = tfidf_vectorizer.transform(text_sub_clusters)

            sub_clusters_top_terms = []
            for vectorized_sub_cluster in vectorized_sub_clusters.todense():
                weights = sorted(enumerate(vectorized_sub_cluster.tolist()[0]), key=itemgetter(1), reverse=True)
                sub_clusters_top_terms.append([(features[token[0]], token[1]) for token in weights[:10]])
            terms = __combine_terms(sub_clusters_top_terms, sub_clusters_size_ratio)
            clusters_labels.append(get_labels(cluster, terms))

    else:
        text_clusters = [" ".join(cluster) for cluster in clusters]
        vectorized_clusters = tfidf_vectorizer.transform(text_clusters)

        clusters_top_terms = []
        for vectorized_cluster in vectorized_clusters.todense():
            weights = sorted(enumerate(vectorized_cluster.tolist()[0]), key=itemgetter(1), reverse=True)
            clusters_top_terms.append([(features[token[0]], token[1]) for token in weights[:10]])

        clusters_labels = [get_labels(cluster, clusters_top_terms[cluster_index]) for cluster_index, cluster in
                           enumerate(clusters)]

    print(evaluate_clusters([[label[0] for label in labels] for labels in clusters_labels]))


def __combine_terms(sub_clusters_top_terms, sub_clusters_size_ratio):
    """Combine terms from sub-clusters"""
    # sub_clusters_top_terms = [cl[:math.ceil(sub_clusters_size_ratio[cli] * 10)] for cli, cl in
    #                          enumerate(sub_clusters_top_terms)]
    # return [term for cl in sub_clusters_top_terms for term in cl]
    # print(sub_clusters_top_terms)

    sub_clusters_size_ratio = [1 + val for val in sub_clusters_size_ratio]
    sub_clusters_top_terms = [[(term[0], term[1] * sub_clusters_size_ratio[cli]) for term in cl] for cli, cl in
                              enumerate(sub_clusters_top_terms)]
    top_terms = {}
    for sub_cluster_top_terms in sub_clusters_top_terms:
        for term in sub_cluster_top_terms:
            if term[0] in top_terms.keys():
                top_terms[term[0]] = top_terms[term[0]] + term[1]
            else:
                top_terms[term[0]] = term[1]

    return Counter(top_terms).most_common(10)


def get_labels(cluster, terms_weights, tokenizer=utils.stem_tokenizer):
    """Generate labels based on terms"""
    noun_phrases = utils.noun_phrases(cluster)
    noun_phrases = utils.filter_terms(noun_phrases, remove_stop_words=False, min_count=0)
    noun_phrases = [noun_phrase[0] for noun_phrase in Counter(noun_phrases).most_common()]  # List of noun_phrases

    # Dict of noun_phrases with key as stemed tokens
    noun_phrases_dict = {}
    for noun_phrase in noun_phrases:
        if tuple(tokenizer(noun_phrase)) not in noun_phrases_dict:
            noun_phrases_dict[tuple(tokenizer(noun_phrase))] = noun_phrase

    # Removed same keys
    terms = [term[0] for term in terms_weights]

    for key_terms, text in dict(noun_phrases_dict).items():
        for key_term in key_terms:
            if key_term not in terms and key_terms in noun_phrases_dict:
                del noun_phrases_dict[key_terms]

    labels = {text: sum([dict(terms_weights)[term] for term in key_terms]) for key_terms, text in
              noun_phrases_dict.items()}
    labels = [label for label in Counter(labels).most_common(10) if label[1] > 0]

    return labels


def sub_clustering(sentences):
    """Sub-clustering of a single cluster"""
    sentences = [sent for sent in sentences if w2v.valid_words(utils.word_tokenizer(sent))]
    sentences_tokens = [w2v.valid_words(utils.word_tokenizer(sent)) for sent in sentences]
    sentences_vectors = [w2v.sum_terms_vector(sent) for sent in sentences_tokens]

    errors = []
    max_sub_clusters = math.floor(len(sentences_vectors) / 5)  # Minimum of 5 sentences on average
    max_sub_clusters = max_sub_clusters if max_sub_clusters <= 5 else 5
    for sub_clusters_count in range(1, max_sub_clusters + 1):  # [1, 2, 3, 4, 5] Number of sub-clusters
        continue_main_loop = False
        sub_clusters_index = __clustering(sentences_vectors, sub_clusters_count)
        for cl in sub_clusters_index:
            if len(cl) < 5:
                continue_main_loop = True
        if continue_main_loop:
            continue

        sub_clusters_vectors = [[sentences_vectors[cli] for cli in cl] for cl in sub_clusters_index]
        errors.append(sum(utils.ClusterAnalysis(cl).mse for cl in sub_clusters_vectors) / sub_clusters_count)
        # errors.append(sum(utils.ClusterAnalysis(cl).sse for cl in sub_clusters_vectors))

    sub_clusters_index = __clustering(sentences_vectors, __optimum_subcluster_count(errors))
    sub_clusters_sentences = [[sentences[cli] for cli in cl] for cl in sub_clusters_index]
    return sub_clusters_sentences


def textrank_labeling(apply_sub_clustering=False):
    """Labeling using textrank"""
    from summa import keywords_custom as keywords
    clusters = list(get_headphone_clusters())
    del clusters[-1]

    if apply_sub_clustering:
        clusters_labels = []
        for cluster in clusters:
            sub_clusters = sub_clustering(cluster)
            sub_clusters_size_ratio = [(len(cl) / len(cluster)) for cl in sub_clusters]
            text_sub_clusters = [" ".join(cl) for cl in sub_clusters]

            sub_clusters_top_terms = []
            for text_sub_cluster in text_sub_clusters:
                terms = keywords.keywords(text_sub_cluster)
                terms = {utils.stem(term): weight for term, weight in terms.items()}
                terms = [(term, weight) for term, weight in terms.items()]
                sub_clusters_top_terms.append(terms[:10])

            terms = __combine_terms(sub_clusters_top_terms, sub_clusters_size_ratio)
            clusters_labels.append(get_labels(cluster, terms))
    else:
        text_clusters = [" ".join(cluster) for cluster in clusters]
        clusters_top_terms = []
        for text_cluster in text_clusters:
            terms = keywords.keywords(text_cluster)
            terms = {utils.stem(term): weight for term, weight in terms.items()}
            terms = [(term, weight) for term, weight in terms.items()]
            clusters_top_terms.append(terms[:10])

        clusters_labels = [get_labels(cluster, clusters_top_terms[cluster_index]) for cluster_index, cluster in
                           enumerate(clusters)]

    print(clusters_labels)
    print(evaluate_clusters([[label[0] for label in labels] for labels in clusters_labels]))


def combined_w2v_labeling(apply_sub_clustering=False):
    """Combine words with w2v for labeling"""
    reviews = get_headphone_reviews()
    count_vectorizer = CountVectorizer(stop_words=STOPWORDS, tokenizer=valid_words_tokenizer)
    count_vectorizer.fit(reviews)
    features = count_vectorizer.get_feature_names()

    # Get cluster of words
    words_vectors = w2v.terms_vectors(features)
    model = AgglomerativeClustering(0.5, "complete", "cosine")
    result = model.fit_predict(words_vectors)
    word_clusters = group_result(result, 2)
    # word_clusters = [[features[word_index] for word_index in cl] for cl in word_clusters]

    df_pipeline = make_pipeline(count_vectorizer, labeling_utils.SimilarDFTransformer(tokens_clusters=word_clusters))
    tf_pipeline = make_pipeline(count_vectorizer, labeling_utils.SimilarTFTransformer(tokens_clusters=word_clusters))

    tfidf_vectorizer = TfidfTransformer()
    tfidf_vectorizer.fit(df_pipeline.transform(reviews))

    clusters = list(get_headphone_clusters())
    del clusters[-1]

    if apply_sub_clustering:
        clusters_labels = []
        for cluster in clusters:
            sub_clusters = sub_clustering(cluster)
            sub_clusters_size_ratio = [(len(cl) / len(cluster)) for cl in sub_clusters]
            text_sub_clusters = [" ".join(cl) for cl in sub_clusters]
            vectorized_sub_clusters = tfidf_vectorizer.transform(tf_pipeline.transform(text_sub_clusters))

            sub_clusters_top_terms = []
            for vectorized_sub_cluster in vectorized_sub_clusters.todense():
                weights = sorted(enumerate(vectorized_sub_cluster.tolist()[0]), key=itemgetter(1), reverse=True)
                sub_clusters_top_terms.append([(features[token[0]], token[1]) for token in weights[:10]])

            terms = __combine_terms(sub_clusters_top_terms, sub_clusters_size_ratio)
            clusters_labels.append(get_labels(cluster, terms))

    else:
        text_clusters = [" ".join(cluster) for cluster in clusters]
        vectorized_clusters = tfidf_vectorizer.transform(tf_pipeline.transform(text_clusters))

        clusters_top_terms = []
        for vectorized_cluster in vectorized_clusters.todense():
            weights = sorted(enumerate(vectorized_cluster.tolist()[0]), key=itemgetter(1), reverse=True)
            clusters_top_terms.append([(features[token[0]], token[1]) for token in weights[:10]])

        clusters_labels = [get_labels(cluster, clusters_top_terms[cluster_index], tokenizer=utils.word_tokenizer) for
                           cluster_index, cluster in
                           enumerate(clusters)]

    print(clusters_labels)
    print(evaluate_clusters([[label[0] for label in labels] for labels in clusters_labels]))


def w2v_multiplied_labeling():
    """w2v multiplied labeling (this method is not giving a good result)"""
    reviews = get_headphone_reviews()
    tfidf_vectorizer = TfidfVectorizer(stop_words=STOPWORDS, tokenizer=valid_words_tokenizer)
    tfidf_vectorizer.fit(reviews)
    features = tfidf_vectorizer.get_feature_names()
    term_vectors = w2v.terms_vectors(features)

    clusters = list(get_headphone_clusters())
    del clusters[-1]
    text_clusters = [" ".join(cluster) for cluster in clusters]
    vectorized_clusters = tfidf_vectorizer.transform(text_clusters)

    clusters_top_terms = []
    for vectorized_cluster in vectorized_clusters.todense():
        weights = vectorized_cluster.tolist()[0]
        indexes_weights = [(term_index, weight) for term_index, weight in enumerate(weights) if weight > 0]

        cluster_vectors = [term_vectors[index_weight[0]] * index_weight[1] for index_weight in indexes_weights]
        distances = utils.ClusterAnalysis(cluster_vectors, distance_metric="euclidean_distances").sorted_distances[:10]
        clusters_top_terms.append([(features[distance_tuple[0]], distance_tuple[1]) for distance_tuple in distances])

    print(clusters_top_terms)
    return
    clusters_labels = [get_labels(cluster, clusters_top_terms[cluster_index]) for cluster_index, cluster in
                       enumerate(clusters)]

    print(evaluate_clusters([[label[0] for label in labels] for labels in clusters_labels]))


def __clustering(sentences_vectors, n_clusters):
    """Clustering of sentences vectors"""
    model = AgglomerativeClustering(n_clusters, "complete", "cosine", "maxclust")
    result = model.fit_predict(sentences_vectors)
    return group_result(result)


def valid_words_tokenizer(text: str) -> List[str]:
    return w2v.valid_words(utils.word_tokenizer(text))


def __optimum_subcluster_count(errors: List[float], threshold=0.8) -> int:
    """Return the optimum number of sub-cluster"""
    _min = min(errors)
    for index, val in enumerate(errors):
        if _min / val >= threshold:
            return index + 1


def evaluate_clusters(clusters_labels):
    clusters_gold_labels = dl.load_json_from_file("../data/headphone_100_aspects.json")
    scores = [evaluate_single_cluster(labels, clusters_gold_labels[cluster_index]) for cluster_index, labels in
              enumerate(clusters_labels)]
    return sum(scores) / len(clusters_labels)


def evaluate_single_cluster(generated_labels: List[str], gold_labels: List[str]):
    return sum(
        max(w2v.similarity(label, gold_label) for gold_label in gold_labels) for label in generated_labels) / len(
        generated_labels)
