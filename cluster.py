import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import Birch
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from scipy.cluster import hierarchy
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import MeanShift
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import FeatureAgglomeration
from hdbscan import HDBSCAN


class MyKmeans:
    def fit(self, data, n_clusters):
        data = np.array(data)
        data = preprocessing.MinMaxScaler().fit_transform(data)
        model = KMeans(n_clusters=n_clusters)
        clustering = model.fit(data)
        return clustering


class MyMiniBatchKMeans:
    def fit(self, data, n_clusters, batch_size):
        data = np.array(data)
        data = preprocessing.MinMaxScaler().fit_transform(data)
        model = MiniBatchKMeans(n_clusters=n_clusters, batch_size=batch_size)
        clustering = model.fit(data)
        return clustering


class MyMeanShift:
    def fit(self, data, bandwidth):
        data = np.array(data)
        data = preprocessing.MinMaxScaler().fit_transform(data)
        model = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        clustering = model.fit(data)
        return clustering


class MyAP:
    def fit(self, data, damping, preference):
        data = np.array(data)
        data = preprocessing.MinMaxScaler().fit_transform(data)
        model = AffinityPropagation(damping=damping, preference=preference)
        clustering = model.fit(data)
        return clustering


class MyBirch:
    def fit(self, data, threshold, branching_factor):
        data = np.array(data)
        data = preprocessing.MinMaxScaler().fit_transform(data)
        model = Birch(threshold=threshold, branching_factor=branching_factor)
        clustering = model.fit(data)
        return clustering


class MyDBSCAN:
    def fit(self, data, eps, min_samples):
        data = np.array(data)
        data = preprocessing.MinMaxScaler().fit_transform(data)
        model = DBSCAN(eps=eps, min_samples=min_samples)
        clustering = model.fit(data)
        return clustering


class MyHDBSCAN:
    def fit(self, data, min_cluster_size, min_samples, alpha, cluster_selection_method):
        data = np.array(data)
        data = preprocessing.MinMaxScaler().fit_transform(data)
        model = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, alpha=alpha,
                        cluster_selection_method=cluster_selection_method, allow_single_cluster=True)
        clustering = model.fit(data)
        return clustering


class MyGMM:
    def fit(self, data, n_clusters, cov_type):
        data = np.array(data)
        data = preprocessing.MinMaxScaler().fit_transform(data)
        model = GaussianMixture(n_components=n_clusters, covariance_type=cov_type)
        clustering = model.fit(data)
        labels = model.predict(data)
        return clustering, labels


class MyHierarchical:
    def fit(self, data, method):
        data = np.array(data)
        data = preprocessing.MinMaxScaler().fit_transform(data)
        distance = hierarchy.distance.pdist(data, 'euclidean')
        linkage = hierarchy.linkage(distance, method=method)
        clustering = hierarchy.fcluster(linkage, t=1, criterion='inconsistent')
        labels = pd.Series(clustering)
        return clustering, labels


class MyAC:
    def fit(self, data, n_clusters):
        data = np.array(data)
        data = preprocessing.MinMaxScaler().fit_transform(data)
        model = AgglomerativeClustering(n_clusters=n_clusters)
        clustering = model.fit(data)
        return clustering


class MySC:
    def fit(self, data, n_clusters):
        data = np.array(data)
        data = preprocessing.MinMaxScaler().fit_transform(data)
        model = SpectralClustering(n_clusters=n_clusters)
        clustering = model.fit(data)
        return clustering


class MyFeatureAgglomeration:
    def fit(self, data, n_clusters, method):
        data = np.array(data)
        data = preprocessing.MinMaxScaler().fit_transform(data)
        model = FeatureAgglomeration(n_clusters=n_clusters, linkage=method)
        clustering = model.fit(data)
        return clustering
