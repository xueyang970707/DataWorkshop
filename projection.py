from sklearn import random_projection, decomposition, manifold, ensemble, preprocessing, discriminant_analysis


class MyGaussianRandomProjection:
    return_data = []

    def __init__(self, source):
        min_max_scaler = preprocessing.MinMaxScaler()
        data_source = min_max_scaler.fit_transform(source)
        pca = random_projection.GaussianRandomProjection(n_components=2)
        self.return_data = pca.fit_transform(data_source)


class MySparseRandomProjection:
    return_data = []

    def __init__(self, source):
        min_max_scaler = preprocessing.MinMaxScaler()
        data_source = min_max_scaler.fit_transform(source)
        pca = random_projection.SparseRandomProjection(n_components=2)
        self.return_data = pca.fit_transform(data_source)


class MyPCA:
    return_data = []

    def __init__(self, source):
        min_max_scaler = preprocessing.MinMaxScaler()
        data_source = min_max_scaler.fit_transform(source)
        pca = decomposition.PCA(n_components=2)
        self.return_data = pca.fit_transform(data_source)


class MyKernelPCA:
    return_data = []

    def __init__(self, source):
        min_max_scaler = preprocessing.MinMaxScaler()
        data_source = min_max_scaler.fit_transform(source)
        pca = decomposition.KernelPCA(n_components=2)
        self.return_data = pca.fit_transform(data_source)


class MySparsePCA:
    return_data = []

    def __init__(self, source):
        min_max_scaler = preprocessing.MinMaxScaler()
        data_source = min_max_scaler.fit_transform(source)
        pca = decomposition.SparsePCA(n_components=2)
        self.return_data = pca.fit_transform(data_source)


class MyRandomizedPCA:
    return_data = []

    def __init__(self, source):
        min_max_scaler = preprocessing.MinMaxScaler()
        data_source = min_max_scaler.fit_transform(source)
        pca = decomposition.RandomizedPCA(n_components=2)
        self.return_data = pca.fit_transform(data_source)


class MyMiniBatchSparsePCA:
    return_data = []

    def __init__(self, source):
        min_max_scaler = preprocessing.MinMaxScaler()
        data_source = min_max_scaler.fit_transform(source)
        pca = decomposition.MiniBatchSparsePCA(n_components=2)
        self.return_data = pca.fit_transform(data_source)


class MyIncrementalPCA:
    return_data = []

    def __init__(self, source):
        min_max_scaler = preprocessing.MinMaxScaler()
        data_source = min_max_scaler.fit_transform(source)
        pca = decomposition.IncrementalPCA(n_components=2)
        self.return_data = pca.fit_transform(data_source)


class MyNMF:
    return_data = []

    def __init__(self, source):
        min_max_scaler = preprocessing.MinMaxScaler()
        data_source = min_max_scaler.fit_transform(source)
        pca = decomposition.NMF(n_components=2)
        self.return_data = pca.fit_transform(data_source)


class MyLDA:
    return_data = []

    def __init__(self, source):
        min_max_scaler = preprocessing.MinMaxScaler()
        data_source = min_max_scaler.fit_transform(source)
        pca = decomposition.LatentDirichletAllocation(n_components=2)
        self.return_data = pca.fit_transform(data_source)


class MyQDA:
    return_data = []

    def __init__(self, source):
        min_max_scaler = preprocessing.MinMaxScaler()
        data_source = min_max_scaler.fit_transform(source)
        pca = discriminant_analysis.QuadraticDiscriminantAnalysis(n_components=2)
        self.return_data = pca.fit_transform(data_source)


class MyFastICA:
    return_data = []

    def __init__(self, source):
        min_max_scaler = preprocessing.MinMaxScaler()
        data_source = min_max_scaler.fit_transform(source)
        pca = decomposition.FastICA(n_components=2)
        self.return_data = pca.fit_transform(data_source)


class MyMDS:
    return_data = []

    def __init__(self, source):
        min_max_scaler = preprocessing.MinMaxScaler()
        data_source = min_max_scaler.fit_transform(source)
        mds = manifold.MDS(n_components=2)
        self.return_data = mds.fit_transform(data_source)


class MyISOMAP:
    return_data = []

    def __init__(self, source):
        min_max_scaler = preprocessing.MinMaxScaler()
        data_source = min_max_scaler.fit_transform(source)
        isomap = manifold.Isomap(n_components=2, n_neighbors=25)
        self.return_data = isomap.fit_transform(data_source)


class MyLLE:
    return_data = []

    def __init__(self, source):
        min_max_scaler = preprocessing.MinMaxScaler()
        data_source = min_max_scaler.fit_transform(source)
        lle = manifold.LocallyLinearEmbedding(n_components=2, n_neighbors=25, method='standard')
        self.return_data = lle.fit_transform(data_source)


class MyLTSA:
    return_data = []

    def __init__(self, source):
        min_max_scaler = preprocessing.MinMaxScaler()
        data_source = min_max_scaler.fit_transform(source)
        lle = manifold.LocallyLinearEmbedding(n_components=2, n_neighbors=25, method='ltsa')
        self.return_data = lle.fit_transform(data_source)


class MyHLLE:
    return_data = []

    def __init__(self, source):
        min_max_scaler = preprocessing.MinMaxScaler()
        data_source = min_max_scaler.fit_transform(source)
        lle = manifold.LocallyLinearEmbedding(n_components=2, n_neighbors=25, method='hessian')
        self.return_data = lle.fit_transform(data_source)


class MyMLLE:
    return_data = []

    def __init__(self, source):
        min_max_scaler = preprocessing.MinMaxScaler()
        data_source = min_max_scaler.fit_transform(source)
        lle = manifold.LocallyLinearEmbedding(n_components=2, n_neighbors=25, method='modified')
        self.return_data = lle.fit_transform(data_source)


class MyTSNE:
    return_data = []

    def __init__(self, source):
        min_max_scaler = preprocessing.MinMaxScaler()
        data_source = min_max_scaler.fit_transform(source)
        tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
        self.return_data = tsne.fit_transform(data_source)


class MySpectralEmbedding:
    return_data = []

    def __init__(self, source):
        min_max_scaler = preprocessing.MinMaxScaler()
        data_source = min_max_scaler.fit_transform(source)
        se = manifold.SpectralEmbedding(n_components=2, random_state=0, eigen_solver='arpack')
        self.return_data = se.fit_transform(data_source)


class MyRandomForestEmbedding:
    return_data = []

    def __init__(self, source):
        min_max_scaler = preprocessing.MinMaxScaler()
        data_source = min_max_scaler.fit(source)
        hasher = ensemble.RandomTreesEmbedding(n_estimators=200, random_state=0, max_depth=None)
        data_transformed = hasher.fit_transform(data_source)
        rfe = decomposition.TruncatedSVD(n_components=2)
        self.return_data = rfe.fit_transform(data_transformed)



