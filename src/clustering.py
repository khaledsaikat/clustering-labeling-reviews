#!/usr/bin/env python3
# Clustering reviews
#
# Load w2v first!!
##

from typing import List

import data_loader as dl
import utils
import word2vec as w2v
from models import AgglomerativeClustering, group_result


def get_kindle_sentences() -> List[str]:
    """Get sentences as a list"""
    reviews = dl.load_json_from_file("../data/kindle_500.json")
    return [sent[1] for review in reviews for sent in enumerate(review)]


def get_headphone_sentences() -> List[str]:
    return dl.load_json_from_file("../data/headphone_sents.json")


def calculate_sse(clusters_sentences_vectors: List[List]):
    return sum(utils.ClusterAnalysis(sentences_vectors).sse for sentences_vectors in clusters_sentences_vectors)


def run_clustering():
    """Running clustering algorithm"""
    sentences = get_headphone_sentences()
    sentences = [sent for sent in sentences if w2v.valid_words(utils.pos_filtered_tokenizer(sent))]
    sentences_tokens = [w2v.valid_words(utils.pos_filtered_tokenizer(sent)) for sent in sentences]
    sentences_vectors = [w2v.sum_terms_vector(sent) for sent in sentences_tokens]

    # ValueError: The condensed distance matrix must contain only finite values. (coming from zero vector)
    model = AgglomerativeClustering(.8, "complete", "cosine")
    result = model.fit_predict(sentences_vectors)
    clusters = group_result(result, 5)
    print(len(clusters))
    # model.dendrogram()

    min_members = 5
    second_phase_indexes = [member for cluster in clusters for member in cluster if len(cluster) >= min_members]
    second_phase_sentences = [sentences[i] for i in second_phase_indexes]
    second_phase_sentences_tokens = [sentences_tokens[i] for i in second_phase_indexes]
    second_phase_vectors = [sentences_vectors[i] for i in second_phase_indexes]

    model = AgglomerativeClustering(5, "complete", "cosine", "maxclust")
    result = model.fit_predict(second_phase_vectors)
    clusters = group_result(result)

    clusters_vectors = [[second_phase_vectors[i] for i in cluster] for cluster in clusters]
    clusters_sentences = [[second_phase_sentences[i] for i in cluster] for cluster in clusters]
    print("ClustersCount:", len(clusters_vectors))
    # print(clusters_sentences)


def evaluate_clustering():
    sentences = get_headphone_sentences()
    # sentences = get_kindle_sentences()
    sentences = [sent for sent in sentences if w2v.valid_words(utils.word_tokenizer(sent))]
    sentences_tokens = [w2v.valid_words(utils.word_tokenizer(sent)) for sent in sentences]
    sentences_vectors = [w2v.sum_terms_vector(sent) for sent in sentences_tokens]
    # print("SSE:", calculate_sse([sentences_vectors]))

    # ValueError: The condensed distance matrix must contain only finite values. (coming from zero vector)
    model = AgglomerativeClustering(.8, "complete", "cosine")
    model = AgglomerativeClustering(10, "complete", "cosine", "maxclust")
    result = model.fit_predict(sentences_vectors)
    clusters = group_result(result, 5)
    print(len(clusters))
    # model.dendrogram()
    # print("Sent Count", len(sentences))
    # clusters_vectors = [[sentences_vectors[i] for i in cluster] for cluster in clusters]
    # print("SSE:", calculate_sse(clusters_vectors))
    # return

    min_members = 5
    second_phase_indexes = [member for cluster in clusters for member in cluster if len(cluster) >= min_members]
    second_phase_vectors = [sentences_vectors[i] for i in second_phase_indexes]

    model = AgglomerativeClustering(10, "complete", "cosine", "maxclust")
    result = model.fit_predict(second_phase_vectors)
    clusters = group_result(result)

    clusters_vectors = [[second_phase_vectors[i] for i in cluster] for cluster in clusters]
    print("Sent Count", len(second_phase_vectors))
    print("SSE:", calculate_sse(clusters_vectors))


def evaluate_manual_clusters():
    data = dl.load_json_from_file("../data/headphone_100_clusters.json")
    clusters = [cluster[0] for cluster in data if cluster[1][0] != "_misc"]
    #print(sum([len(cl) for cl in clusters]))

    ssse = 0
    for cluster in clusters:
        sentences = [sent for sent in cluster if w2v.valid_words(utils.word_tokenizer(sent))]
        sentences_tokens = [w2v.valid_words(utils.word_tokenizer(sent)) for sent in sentences]
        sentences_vectors = [w2v.sum_terms_vector(sent) for sent in sentences_tokens]
        # print(utils.ClusterAnalysis(sentences_vectors).mse)
        ssse += utils.ClusterAnalysis(sentences_vectors).sse

    print("SSE:", ssse)
    # print(clusters)


if __name__ == "__main__":
    evaluate_manual_clusters()
