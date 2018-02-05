#!/usr/bin/env python3
import word2vec as w2v
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob


class Evaluation:
    """Similarities between two strings"""

    index_vector = []
    first_vector = []
    second_vector = []

    def __init__(self, first, second):
        self.first = TextBlob(first)
        self.second = TextBlob(second)
        self.generate_vectors()

    @staticmethod
    def string_to_vec(text):
        return list(w2v.combined_terms_vector(text.words))

    def generate_vectors(self):
        self.first = [self.string_to_vec(self.first)]
        self.second = [self.string_to_vec(self.second)]

    def similarity(self):
        return cosine_similarity(self.first, self.second)[0][0]


def run():
    data = {"sound quality": "Good sound",
            "quality": "These by far are the best headphones I have ever had.",
            "noise cancellation": "These panasonic keep out exterior noise pretty well, too.",
            "comfortability": "comfortable to wear and they do not hurt your ears.",
            "design": "But with this unique ergo design, it addresses that problem.",
            "fit": "Sound good and fit comfortably.",
            "color": "Plus comes in a nice variety of colors."}

    for k,v in data.items():
        print(Evaluation(k,v).similarity(), k, ",", v)


if __name__ == "__main__":
    run()

