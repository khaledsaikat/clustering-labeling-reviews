#!/usr/bin/env python3

##
# Helping script for manual annotation
##
import json
from collections import Counter
from pprint import pprint

#from nltk.stem import PorterStemmer

#PS = PorterStemmer()


def load_file(file_path):
    """Loading a json file"""
    with open(file_path) as json_data:
        return json.load(json_data)


def write_file(data, file_path):
    """Writing data to a json file"""
    with open(file_path, "w") as json_file:
        json.dump(data, json_file)


class Annotation:
    reviews = []

    def __init__(self, file_prefix, base_path=""):
        self.base_path = base_path
        self.file_prefix = file_prefix
        self._load_reviews(self.base_path + self.file_prefix + ".json")

    def _load_reviews(self, file_path):
        """Load reviews from a json file"""
        self.reviews = load_file(file_path)

    def get_sentences(self):
        """Collect all sentences with aspects. Return list of tuple"""
        return [sentence for review in self.reviews for sentence in review.items()]

    def show_manual_aspects(self):
        """Showing manually annotated aspects with counting"""
        aspects = self.get_manual_aspects()
        print(Counter(aspects))
        print(len(Counter(aspects)))

    def get_manual_aspects(self):
        """Get all aspects name with number of reviews where the aspect were found"""
        aspects = []
        [aspects.extend(sentence[1]) for review in self.reviews for sentence in review.items() if len(sentence[1]) > 0]
        return [aspect.lower() for aspect in aspects]

    def aspects_to_clusters(self):
        """Cluster sentences based on aspects"""
        aspects_clusters = load_file(self.base_path + self.file_prefix + "_aspects.json")
        clusters = {tuple(aspects): [] for aspects in aspects_clusters}
        clusters[("_misc",)] = []
        sentences = self.get_sentences()
        for sentence in sentences:
            aspect_found = False
            if len(sentence[1]) > 0:
                for sent_aspect in sentence[1]:
                    if aspect_found:
                        continue
                    for aspects in aspects_clusters:
                        if sent_aspect in aspects:
                            clusters[tuple(aspects)].append(sentence[0])
                            aspect_found = True
                if not aspect_found:
                    clusters[("_misc",)].append(sentence[0])
            else:
                clusters[("_misc",)].append(sentence[0])

        clusters_counts = {aspects: len(sentences) for aspects, sentences in clusters.items()}
        print("ClusterSentCount:")
        pprint(Counter(clusters_counts))
        print("SentCounts:", len(sentences), sum(counts for counts in clusters_counts.values()))
        print("ClusterCount:", len(clusters))
        #self._write_clusters(clusters)

    def _write_clusters(self, clusters):
        """Write clusters to the file"""
        clusters = [[sentences, aspects] for aspects, sentences in clusters.items()]
        write_file(clusters, self.base_path + self.file_prefix + "_temp.json")


def full_reviews_to_dict_aspects(input_path, optput_path="headphone_100.json"):
    """Covert old style annotation to new style"""
    reviews = json.load(open(input_path))
    reviews = [_review_to_to_dict_aspects(review) for review in reviews]
    json.dump(reviews, open(optput_path, "w"))


def _review_to_to_dict_aspects(review):
    return {line: _sentence_index_to_aspects(review["aspects"], int(idx)) for idx, line in
            review["reviewTextJson"].items()}


def _sentence_index_to_aspects(aspects, _idx):
    indexes = set([idx for name, indexes in aspects.items() for idx in indexes])
    indexes = {idx: [] for idx in indexes}
    for name, idxs in aspects.items():
        for idx in idxs:
            indexes[idx].append(name)
    return indexes[_idx] if _idx in indexes else []


if __name__ == "__main__":
    annotation = Annotation("kindle_500", "../data/")
    annotation.show_manual_aspects()
    #annotation.aspects_to_clusters()
