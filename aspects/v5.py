#!/usr/bin/env python3
# Clustering reviews
#
# Load w2v first!!
##

import json
from collections import Counter
from itertools import chain
from typing import List

import data_loader as dl
import gensim
import numpy as np
import utils
import word2vec as w2v
from cluster import AgglomerativeClustering, group_result
from sklearn.metrics.pairwise import cosine_distances
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk

Sentence = str


def load_reviews_sentences_by_line(file_path: str, max_items: int = 100):
    """Load reviews sentences from a json file and return a generator."""
    with open(file_path) as json_file:
        item_count = 0
        for line in json_file:
            text = json.loads(line)["reviewText"]
            for sent in nltk.sent_tokenize(text, "english"):
                item_count += 1
                if item_count > max_items:
                    return
                yield sent


def load_reviews_by_line(file_path: str, max_lines: int = 100):
    """Load reviews from a json file and return a generator."""
    with open(file_path) as json_file:
        line_count = 0
        for line in json_file:
            line_count += 1
            if line_count > max_lines:
                break
            yield json.loads(line)["reviewText"]


def important_words(sentences: List[Sentence], min_percent=0.005):
    """Get important words based on term frequency and w2v similarity"""
    words = Counter((word for sent in sentences for word in utils.pos_filtered_tokenizer(sent, ["NN"])))
    min_count = sum(words.values()) * min_percent
    print(min_count)
    valid_words = list(w2v.valid_words(words.keys()))
    word_vectors = np.array(list(w2v.terms_vectors(valid_words)))
    clusters = cluster_words(word_vectors, valid_words)
    return [w for cluster in clusters for w in cluster if sum(words[w] for w in cluster) >= min_count]


def cluster_words(word_vectors, words):
    """Cluster words from their w2v vectors"""
    ac = AgglomerativeClustering(.5, "complete", "cosine")
    result = ac.fit_predict(word_vectors)
    return group_result(result, index_names=words)


def filter_sentences(sentences: List[Sentence]):
    words = set(important_words(sentences, 0.002))
    s = [sent for sent in sentences if words.intersection(utils.raw_word_tokenizer(sent.lower()))]
    print(set(sentences).difference(s))
    print(len(sentences) - len(s))


class TaggedDocumentsSource:
    def __init__(self, source, index_prefix=""):
        """source as path or List[Sentence]"""
        self.source = load_reviews_by_line(source) if type(source) is str else source
        self.index_prefix = index_prefix

    def __iter__(self):
        for index, doc in enumerate(self.source):
            index = self.index_prefix + str(index) if self.index_prefix else index
            yield gensim.models.doc2vec.TaggedDocument(utils.pos_filtered_tokenizer(doc), [index])


def get_review_sample() -> List[Sentence]:
    return dl.loadJsonFromFile("../data/headphone_sents.json")


def train_doc2vec(iterator_source):
    model = gensim.models.Doc2Vec(size=300, min_count=0, alpha=0.025, min_alpha=0.025)
    model.build_vocab(iterator_source)
    # training of model
    for epoch in range(100):
        model.train(iterator_source, total_examples=489)
        model.alpha -= 0.002
        model.min_alpha = model.alpha
        model.train(iterator_source, total_examples=489)

    return model


def clustering():
    from cluster import AgglomerativeClustering, group_result
    import numpy as np
    data = get_review_sample()
    # model = train_doc2vec(TaggedDocumentsSource(data))
    # model = gensim.models.Doc2Vec(TaggedDocumentsSource(data), size=300, min_count=0, iter=10)
    # model.save("../data/models/headphone_clusters.model")
    model = gensim.models.doc2vec.Doc2Vec.load("../data/models/headphone_clusters.model")

    """
    ac = AgglomerativeClustering(.5, "complete", "cosine")
    ac.fit_predict(np.array(model.docvecs))
    ac.dendrogram("complete cosine")

    ac = AgglomerativeClustering(.5, "ward")
    ac.fit_predict(np.array(model.docvecs))
    ac.dendrogram("ward")

    ac = AgglomerativeClustering(.5, "single")
    ac.fit_predict(np.array(model.docvecs))
    ac.dendrogram("single")

    ac = AgglomerativeClustering(.5, "complete")
    ac.fit_predict(np.array(model.docvecs))
    ac.dendrogram("complete")

    ac = AgglomerativeClustering(.5, "single", "cosine")
    ac.fit_predict(np.array(model.docvecs))
    ac.dendrogram("single cosine")

    ac = AgglomerativeClustering(.5, "complete", "cosine")
    ac.fit_predict(np.array(model.docvecs))
    ac.dendrogram("complete cosine")

    return
    """

    # ac = AgglomerativeClustering(0.5, "single", "cosine")
    ac = AgglomerativeClustering(10, criterion="maxclust")
    res = ac.fit_predict(np.array(model.docvecs))
    print(res)
    print(group_result(res))
    print("GroupCount", len(group_result(res)))
    print(ac.linkage_matrix)
    ac.dendrogram()

    for i in group_result(res)[0]:
        print(data[i])

    return

    i = 0
    print(data[i])
    similar = model.docvecs.most_similar(i)
    for s in similar:
        print(data[s[0]], s[1])


def test_clustering():
    from cluster import AgglomerativeClustering
    import numpy as np
    data = ["sound quality", "good sound quality", "very good sound quality", "bad sound", "bad quality",
            "hello world", "nice world", "beautiful day", "big boy", "little cat", "row file", "sound quality"]
    model = gensim.models.Doc2Vec(TaggedDocumentsSource(data), size=300, min_count=0, iter=100)
    ac = AgglomerativeClustering(.5, "complete")
    ac.fit_predict(np.array(model.docvecs))
    ac.dendrogram("complete cosine")


def tfidf_vectorizer(sentences: List[str]):
    sentences = [" ".join(utils.stem_tokenizer(sent)) for sent in sentences]
    vectorizer = TfidfVectorizer(stop_words="english")
    #print(vectorizer.fit_transform(sentences).todense().shape)
    return vectorizer.fit_transform(sentences).todense()


def run_tfidf_clustering():
    sentences = get_review_sample()
    sentences = [" ".join(utils.pos_stem_tokenizer(sent)) for sent in sentences]
    sentences = [sent for sent in sentences if sent]
    sentences_vectors = TfidfVectorizer().fit_transform(sentences).todense()

    model = AgglomerativeClustering(1.5)
    result = model.fit_predict(sentences_vectors)
    clusters = group_result(result)
    print(clusters)
    print(len(clusters))

    for sent_index in clusters[124]:
        print(sentences[sent_index])

    model.dendrogram()


def run_w2v_clustering():
    sentences = get_review_sample()
    #sentences = load_reviews_sentences_by_line("/Users/khaled/Downloads/reviews_Electronics.json", 5000)
    sentences = [sent for sent in sentences if w2v.valid_words(utils.pos_filtered_tokenizer(sent))]
    sentences_tokens = [w2v.valid_words(utils.pos_filtered_tokenizer(sent)) for sent in sentences]
    sentences_vectors = [w2v.sum_terms_vector(sent) for sent in sentences_tokens]
    print(len(sentences_vectors))

    # ValueError: The condensed distance matrix must contain only finite values. (coming from zero vector)
    model = AgglomerativeClustering(.5, "complete", "cosine")
    result = model.fit_predict(sentences_vectors)
    clusters = group_result(result, 5)
    print(len(clusters))
    model.dendrogram()

    #print([sentences_tokens[m] for c in clusters for m in c if len(c) < 5])
    #return

    min_members = 5
    second_phase_indexes = [member for cluster in clusters for member in cluster if len(cluster) >= min_members]
    second_phase_sentences = [sentences_tokens[i] for i in second_phase_indexes]
    second_phase_vectors = [sentences_vectors[i] for i in second_phase_indexes]
    print(len(second_phase_vectors))
    model = AgglomerativeClustering(10, "complete", "cosine", "maxclust")
    result = model.fit_predict(second_phase_vectors)
    clusters = group_result(result)

    #kmeans = KMeans(n_clusters=10).fit(second_phase_vectors)
    #clusters = group_result(kmeans.labels_)

    print(len(clusters))
    model.dendrogram()

    #dl.writeJsonToFile([[sentences_tokens[sent_id] for sent_id in cluster] for cluster in clusters], "kindle_500_struct.json")


def run_filter_sentences():
    data = get_review_sample()
    # w = important_words(data, 5)
    filter_sentences(data)


def run_combined_doc2vec():
    raw_source = "/Users/khaled/Downloads/reviews_Electronics.json"
    sentences = get_review_sample()
    test_sent = ["Good sound quality", "Bad sound quality", "Different varieties of colors"]
    tagged_documents = chain(TaggedDocumentsSource(load_reviews_sentences_by_line(raw_source, 10000000)),
                             TaggedDocumentsSource(sentences, "_"), TaggedDocumentsSource(test_sent, "__"))
    #tagged_documents = chain(TaggedDocumentsSource(sentences), TaggedDocumentsSource(test_sent, "__"))
    model = gensim.models.Doc2Vec(tagged_documents, size=100, min_count=0, iter=20, window=8, workers=4)
    print(cosine_distances([model.docvecs["__0"]], [model.docvecs["__1"]]))
    print(cosine_distances([model.docvecs["__0"]], [model.docvecs["__2"]]))

    return
    vecs = [model.docvecs["_"+str(i)] for i in range(len(sentences))]
    ac = AgglomerativeClustering(.5, "complete", "cosine")
    ac.fit_predict(np.array(vecs))
    ac.dendrogram("complete cosine")
    #return model


if __name__ == "__main__":
    #run_combined_doc2vec()
    #tfidf_vectors(get_review_sample())
    run_w2v_clustering()
