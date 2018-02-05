#!/usr/bin/env python3
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline


def document_frequency(X):
    """Count the number of non-zero values for each feature in sparse X."""
    if sp.isspmatrix_csr(X):
        return np.bincount(X.indices, minlength=X.shape[1])
    else:
        return np.diff(sp.csc_matrix(X, copy=False).indptr)


def idf(X, smooth_idf=True):
    n_samples, n_features = X.shape

    df = document_frequency(X)
    # perform idf smoothing if required
    df += int(smooth_idf)
    n_samples += int(smooth_idf)
    return np.log(float(n_samples) / df) + 1.0


def combined_document_frequency(X, indices=[]):
    """Get combined df for provided indeces"""
    n_samples, n_features = X.shape
    summed = csr_matrix(np.zeros(n_samples).reshape(n_samples, 1))
    for i in indices:
        summed += X.getcol(i)

    return sum(np.bincount(summed.getcol(0).indices))


corpus = [
    'This is the first document .',
    'This is document the second document document1 collection.',
    'And the third one in collection document1.',
    'Is this the first document document1?',
]

corpus_test = [
    "another test document ",
    "test document for second time document collection",
    "This is a final one"
]

"""Testing TfidfVectorizer fit and transform for train and test corpus"""
tv = TfidfVectorizer(stop_words="english")
tv.fit(corpus)
print(tv.get_feature_names())
print("IDF:", tv.idf_)
# print(tv.transform(corpus).toarray())

cv = CountVectorizer(stop_words="english")
cv.fit(corpus)
tfa = cv.transform(corpus).toarray()

print(tfa)
df = document_frequency(cv.transform(corpus))
print(df)
print(combined_document_frequency(cv.transform(corpus), [1, 2, 3]))
idf = idf(cv.transform(corpus))
print(idf)

tfidfa = tfa * tv.idf_
print(tfidfa)
# print(preprocessing.normalize(tfidfa, norm='l2'))




quit()


def testing_TfidfVectorizer_fit_transform():
    """Testing TfidfVectorizer fit and transform for train and test corpus"""
    tv = TfidfVectorizer(stop_words="english")
    tv.fit(corpus)
    print(tv.get_feature_names())
    print("IDF:", tv.idf_)
    print(tv.transform(corpus_test).toarray())

    cv = CountVectorizer(stop_words="english")
    cv.fit(corpus)
    tfa = cv.transform(corpus_test).toarray()
    tfidfa = tfa * tv.idf_
    print(tfidfa)
    print(preprocessing.normalize(tfidfa, norm='l2'))


def example_Vectorizer_Transformer():
    vectorizer = CountVectorizer(stop_words="english")
    X_train_counts = vectorizer.fit_transform(corpus)
    print("Count:\n", X_train_counts.toarray())

    tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
    X_train_tf = tf_transformer.transform(X_train_counts)
    print("tf:\n", X_train_tf.toarray())
    # print(TfidfTransformer(use_idf=False).fit_transform(X_train_counts).toarray())

    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    print("idf:\n", tfidf_transformer.idf_)
    print("tfidf:\n", X_train_tfidf.toarray())
    # print(TfidfTransformer().fit_transform(X_train_counts).toarray())

    # data = TfidfVectorizer(stop_words="english").fit_transform(corpus)
    # print(data.toarray())


def example_pipeline():
    pipeline = Pipeline([
        ("vect", CountVectorizer(stop_words="english")),
        ("tfidf", TfidfTransformer())
    ])
    print("Pipeline:\n", pipeline.fit_transform(corpus).toarray())
    print("TfidfVectorizer:\n", TfidfVectorizer(stop_words="english").fit_transform(corpus).toarray())


"""
# test clustering
import data_loader as dl
from sklearn.feature_extraction.text import TfidfVectorizer
import cluster
data = dl.loadJsonFromFile("../data/headphone_clusters.json")
docs = list(set([doc for cl in data.values() for doc in cl]))
vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 1))
X = vectorizer.fit_transform(docs)

ca = cluster.AgglomerativeClustering(1.9)
r = ca.fit_predict(X.toarray())

#kmeans = KMeans(n_clusters=10, random_state=0).fit(X)
#c=cluster.group_clustering_result(kmeans.labels_)
c = cluster.group_clustering_result(r)

[print("\n", ci) for ci in c]

[print(i,":\n",docs[i]) for i in [3, 34]]
"""
