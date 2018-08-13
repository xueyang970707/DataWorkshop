# encoding:utf-8
import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics

'''
        if clusters > 0:
            model = KMeans(n_clusters=clusters)
            clustering = model.fit(data)
        else:
            k_bound = 0
            n_samples, n_features = data.shape
            if n_samples > 11:
                k_bound = 11
            else:
                k_bound = n_samples
            score_list = []
            k_list = []
            for k in range(2, k_bound):
                model = KMeans(n_clusters=k)
                clustering = model.fit_predict(data)
                score = metrics.calinski_harabaz_score(data, clustering)
                score_list.append(score)
                k_list.append(k)
            index = score_list.index(max(score_list))
            model = KMeans(n_clusters=k_list[index])
            clustering = model.fit(data)
'''