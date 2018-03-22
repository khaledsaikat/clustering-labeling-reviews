from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

"""
Cluster words:
words = ['noise', 'sound', 'loud', 'cancel', 'canceling', 'cancellation', 'headphones', 'earbuds', 'earbud', 'ipod',
        'cord', 'cable', 'cords', 'cables', 'jacket', 'cheap', 'price', 'color', 'plastic', 'tangle']
X = np.array([model[w] for w in words])
ac = AgglomerativeClustering(0.5, 'complete', 'cosine')
group_result(ac.fit_predict(X))
"""


class AgglomerativeClustering:
    """
    Hierarchical Agglomerative Clustering

    Recursively merges the pair of clusters that minimally increases
    a given linkage distance.

    We are not using sklearn.cluster.AgglomerativeClustering because it requires n_clusters parameter

    For more details of parameters, see documentation of scipy.cluster.hierarchy.linkage and scipy.cluster.hierarchy.linkage.fcluster

    https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.fcluster.html
    """

    def __init__(self, criterion_threshold, linkage_method="ward", distance_metric="euclidean", criterion="distance"):
        self.criterion_threshold = criterion_threshold
        self.criterion = criterion
        self.distance_metric = distance_metric
        self.linkage_method = linkage_method
        self.linkage_matrix_ = None

    def fit_predict(self, X):
        """Performs clustering on X and returns cluster labels.

        :param X: ndarray, shape (n_samples, n_features) Input data.
        :return ndarray, shape (n_samples,) cluster labels
        """
        # generate the linkage matrix
        self.linkage_matrix_ = linkage(X, self.linkage_method, self.distance_metric)
        return fcluster(self.linkage_matrix_, self.criterion_threshold, criterion=self.criterion)

    @property
    def linkage_matrix(self):
        return self.linkage_matrix_

    def dendrogram(self, title="HAC Dendrogram"):
        from matplotlib import pyplot as plt
        """Showing dendrogram"""
        plt.title(title)
        plt.xlabel('index')
        plt.ylabel('distance')
        dendrogram(self.linkage_matrix)
        plt.axhline(y=self.criterion_threshold, c="k")
        plt.show()

    @staticmethod
    def example():
        import numpy as np
        X = np.array([(3, 5), (3, 4), (5, 7), (3, 5), (6, 4)])
        ac = AgglomerativeClustering(3.5)
        res = ac.fit_predict(X)
        print(res)
        print(group_result(res))
        print(ac.linkage_matrix)
        ac.dendrogram()


def group_result(result, n_min_item=0, index_names=[]):
    """Grouping result of clustering algorithm"""
    groups = [[] for i in range(max(result) + 1)]
    for index, value in enumerate(result):
        groups[value].append(index)
    if index_names:
        result = [[index_names[index] for index in group] for group in groups[1:]]
    else:
        result = groups[1:]
    if n_min_item > 1:
        result = [items for items in result if len(items) >= n_min_item]
    return result


if __name__ == "__main__":
    AgglomerativeClustering.example()
