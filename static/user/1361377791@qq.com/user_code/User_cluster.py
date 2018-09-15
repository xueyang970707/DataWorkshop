import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.cluster import KMeans
from clusters import ClusterWay
global labels
data = np.array(final_data_object['no_identifiers_data_list'])
data = preprocessing.MinMaxScaler().fit_transform(data)
print("hello,it's kmeans")
model = KMeans(n_clusters=4)
clustering = model.fit(data)
labels = clustering.labels_
labels = labels.tolist()