from sklearn import random_projection, decomposition, manifold, ensemble, preprocessing, discriminant_analysis
global User_data
#User_data=data%2

min_max_scaler = preprocessing.MinMaxScaler()
data_source = min_max_scaler.fit_transform(final_data_object['no_identifiers_data_list'])
pca = decomposition.PCA(n_components=2)
a = pca.fit_transform(data_source)
User_data=a.tolist()