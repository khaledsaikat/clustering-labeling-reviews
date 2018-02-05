#!/usr/bin/env python3

import json
from typing import List

import gensim
import utils
import data_loader as dl

Sentence = str

DOC2VEC_MODEL = None


def load_reviews_by_line(file_path: str, max_lines: int = 2000000):
    """Load reviews from a json file and return a generator."""
    with open(file_path) as json_file:
        line_count = 0
        for line in json_file:
            line_count += 1
            if line_count > max_lines:
                break
            yield json.loads(line)["reviewText"]


class TaggedDocumentsSource:
    def __init__(self, source, index_prefix=""):
        """source as path or List[Sentence]"""
        self.source = load_reviews_by_line(source) if type(source) is str else source
        self.index_prefix = index_prefix

    def __iter__(self):
        for index, doc in enumerate(self.source):
            index = self.index_prefix + str(index) if self.index_prefix else index
            yield gensim.models.doc2vec.TaggedDocument(utils.word_tokenizer(doc), [index])


def base_train():
    file_source = "/Users/khaled/Downloads/reviews_Electronics.json"
    model = gensim.models.Doc2Vec(TaggedDocumentsSource(file_source), size=300, min_count=0, iter=1000)
    model.save("../data/models/doc2vec-temp.model")
    print(len(model.docvecs))


def train_sentences(sentences: List[Sentence]):
    global DOC2VEC_MODEL
    DOC2VEC_MODEL.build_vocab(TaggedDocumentsSource(sentences))


def load_base_model():
    global DOC2VEC_MODEL
    path = "../data/models/doc2vec.model"
    DOC2VEC_MODEL = gensim.models.doc2vec.Doc2Vec.load(path)


def run():
    train_sentences(get_review_sample())
    # model = load_model("../data/models/doc2vec.model")
    # print(model.docvecs.most_similar(1))


def get_review_sample() -> List[Sentence]:
    data = dl.loadJsonFromFile("../data/headphone_clusters.json")
    # return data["quality"]
    return [sent for cluster in data.values() for sent in cluster]


def clustering():
    from cluster import AgglomerativeClustering, group_result
    import numpy as np
    data = get_review_sample()

    for i in [160, 192, 251, 269, 335, 362]:
        print(data[i])
    return

    #model = gensim.models.Doc2Vec(TaggedDocumentsSource(data), size=300, min_count=2, iter=1000)
    #model.save("../data/models/headphone_clusters.model")
    model = gensim.models.doc2vec.Doc2Vec.load("../data/models/headphone_clusters.model")

    ac = AgglomerativeClustering(0.5, "complete", "cosine")
    res = ac.fit_predict(np.array(model.docvecs))
    print(res)
    print(group_result(res))
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



# file_source = "/Users/khaled/Downloads/reviews_Electronics.json"
# data = load_reviews_by_line(file_source)
# print(len(list(data)))

# print(list(TaggedDocumentsSource(file_source)))

if __name__ == "__main__":
    clustering()
