from flask import Flask, request, json, redirect, render_template, url_for, flash
from statistic import Statistic
from projection import MyGaussianRandomProjection, MySparseRandomProjection, \
    MyPCA, MyMiniBatchSparsePCA, MyKernelPCA, MySparsePCA, MyRandomizedPCA, MyIncrementalPCA, \
     MyNMF, MyLDA, MyFastICA, MyMDS, MyISOMAP, MyLLE, MyHLLE, MyMLLE, MyLTSA, MyTSNE, MySpectralEmbedding, MyRandomForestEmbedding
from cluster import MyAP, MyBirch, MyDBSCAN, MyGMM, MyHierarchical, MyKmeans, MyMeanShift, MySC, MyAC,\
    MyMiniBatchKMeans, MyHDBSCAN, MyFeatureAgglomeration
import ast
import numpy as np

app = Flask(__name__)

global source_data
global data_frame
source_data = []
data_frame = []

global dictionary_data
global dictionary_data_title
dictionary_data = []
dictionary_data_title = []


def data_list_to_dictionary(list_key, list_value):
    if len(list_key) != len(list_value):
        print("键值对的长度不匹配")
        exit(0)
    dict = {}
    for i in range(len(list_key)):
        dict[list_key[i]] = list_value[i]
    return dict


@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST', 'GET'])
def upload():
    global source_data
    global data_frame
    global dictionary_data
    global dictionary_data_title
    if request.method == 'POST':
        source_data = []
        json_data = request.form.get('d')
        temp = json.loads(json_data)
        data_frame = temp[0]
        del temp[0]
        del temp[-1]
        for temp_list in temp:
            sample_list = []
            for str in temp_list:
                t = ast.literal_eval(str)
                sample_list.append(t)
            source_data.append(sample_list)
        for l in temp:
            list_num = []
            for num in l:
                list_num.append(num)
            my_dic = data_list_to_dictionary(data_frame, list_num)
            dictionary_data.append(my_dic)
        scatter_title = data_list_to_dictionary(data_frame, data_frame)
        dictionary_data_title.append(scatter_title)
    return render_template('Upload.html')


@app.route('/upload/clean', methods=['POST', 'GET'])
def upload_clean():
    return render_template('parallel_coordinates.html', data=source_data, title=data_frame, method='Parallel Coordinate')


@app.route('/upload/statistics', methods=['POST', 'GET'])
def upload_statistics():
    stt = Statistic(source_data, data_frame)
    mode = stt.mode[0][0]
    mode = mode.tolist()
    return render_template('upload_statistics.html', mean=stt.mean, median=stt.median, mode=mode, min=stt.min, max=stt.max,
                           var=stt.var, frame=data_frame, method='Statistics')


@app.route('/upload/clusters', methods=['POST', 'GET'])
def upload_clusters():
    return render_template("upload_clusters.html", method='Hierarchy')


@app.route('/upload/projection', methods=['POST', 'GET'])
def upload_projection():
    global dictionary_data
    model = MyTSNE(source_data)
    projection_data = model.return_data
    projection_data = projection_data.tolist()
    return render_template("draw_projection.html", data=projection_data, data_obj=dictionary_data, method='TSNE')


@app.route('/upload/visualization', methods=['POST', 'GET'])
def upload_visualization():
    global data_frame
    global dictionary_data
    return render_template("radviz.html", data=dictionary_data, title=data_frame, data_obj= dictionary_data, method='Radviz')


@app.route('/statistics')
def statistics():
    return render_template('Statistics.html')


@app.route('/statistics/statistic')
def statistic():
    stt = Statistic(source_data, data_frame)
    mode = stt.mode[0][0]
    mode = mode.tolist()
    return render_template('draw_statistics.html', mean=stt.mean, median=stt.median, mode=mode, min=stt.min, max=stt.max,
                           var=stt.var, cov=stt.cov, data=source_data, frame=data_frame, method='Statistics')


@app.route('/projection')
def projection():
    return render_template('projection.html')


@app.route('/projection/grp')
def grp():
    model = MyGaussianRandomProjection(source_data)
    projection_data = model.return_data
    projection_data = projection_data.tolist()
    return render_template("draw_projection.html", data=projection_data, data_obj=dictionary_data, method='GaussianRandomProjection')


@app.route('/projection/srp')
def srp():
    model = MySparseRandomProjection(source_data)
    projection_data = model.return_data
    projection_data = projection_data.tolist()
    return render_template("draw_projection.html", data=projection_data, data_obj=dictionary_data, method='SparseRandomProjection')


@app.route('/projection/pca')
def pca():
    model = MyPCA(source_data)
    projection_data = model.return_data
    projection_data = projection_data.tolist()
    return render_template("pca.html", data=projection_data, data_obj=dictionary_data, method='PCA')


@app.route('/projection/kpca')
def kpca():
    model = MyKernelPCA(source_data)
    projection_data = model.return_data
    projection_data = projection_data.tolist()
    return render_template("draw_projection.html", data=projection_data, data_obj=dictionary_data, method='KernelPCA')


@app.route('/projection/spca')
def spca():
    model = MySparsePCA(source_data)
    projection_data = model.return_data
    projection_data = projection_data.tolist()
    return render_template("draw_projection.html", data=projection_data, data_obj=dictionary_data, method='SparsePCA')


@app.route('/projection/rpca')
def rpca():
    model = MyRandomizedPCA(source_data)
    prokection_data = model.return_data
    prokection_data = prokection_data.tolist()
    return render_template("draw_projection.html", data=prokection_data, data_obj=dictionary_data, method='RandomizedPCA')


@app.route('/projection/minibatchsparsepca')
def minibatchsparsepca():
    model = MyMiniBatchSparsePCA(source_data)
    projection_data = model.return_data
    projection_data = projection_data.tolist()
    return render_template("draw_projection.html", data=projection_data, data_obj=dictionary_data, method='MiniBatchSparsePCA')


@app.route('/projection/ipca')
def ipca():
    model = MyIncrementalPCA(source_data)
    projection_data = model.return_data
    projection_data = projection_data.tolist()
    return render_template("draw_projection.html", data=projection_data, data_obj=dictionary_data, method='IncrementalPCA')


@app.route('/projection/nmf')
def nmf():
    model = MyNMF(source_data)
    projection_data = model.return_data
    projection_data = projection_data.tolist()
    return render_template("draw_projection.html", data=projection_data, data_obj=dictionary_data, method='NMF')


@app.route('/projection/lda')
def lda():
    model = MyLDA(source_data)
    projection_data = model.return_data
    projection_data = projection_data.tolist()
    return render_template("draw_projection.html", data=projection_data, data_obj=dictionary_data, method='LDA')


@app.route('/projection/fica')
def fica():
    model = MyFastICA(source_data)
    projection_data = model.return_data
    projection_data = projection_data.tolist()
    return render_template("draw_projection.html", data=projection_data, data_obj=dictionary_data, method='FastICA')


@app.route('/projection/mds')
def mds():
    model = MyMDS(source_data)
    projection_data = model.return_data
    projection_data = projection_data.tolist()
    return render_template("draw_projection.html", data=projection_data, data_obj=dictionary_data, method='MDS')


@app.route('/projection/isomap')
def isomap():
    model = MyISOMAP(source_data)
    projection_data = model.return_data
    projection_data = projection_data.tolist()
    return render_template("draw_projection.html", data=projection_data, data_obj=dictionary_data, method ='ISOMAP')


@app.route('/projection/lle')
def lle():
    model = MyLLE(source_data)
    projection_data = model.return_data
    projection_data = projection_data.tolist()
    return render_template("draw_projection.html", data=projection_data, data_obj=dictionary_data, method ='LLE')


@app.route('/projection/hlle')
def hlle():
    model = MyHLLE(source_data)
    projection_data = model.return_data
    projection_data = projection_data.tolist()
    return render_template("draw_projection.html", data=projection_data, data_obj=dictionary_data, method='HLLE')


@app.route('/projection/mlle')
def mlle():
    model = MyMLLE(source_data)
    projection_data = model.return_data
    projection_data = projection_data.tolist()
    return render_template("draw_projection.html", data=projection_data, data_obj=dictionary_data, method='MLLE')


@app.route('/projection/ltsa')
def ltsa():
    model = MyLTSA(source_data)
    projection_data = model.return_data
    projection_data = projection_data.tolist()
    return render_template("draw_projection.html", data=projection_data, data_obj=dictionary_data, method='LTSA')


@app.route('/projection/tsne')
def tsne():
    model = MyTSNE(source_data)
    projection_data = model.return_data
    projection_data = projection_data.tolist()
    return render_template("draw_projection.html", data=projection_data, data_obj=dictionary_data, method='TSNE')


@app.route('/projection/spectralembedding')
def spectralembedding():
    model = MySpectralEmbedding(source_data)
    projection_data = model.return_data
    projection_data = projection_data.tolist()
    return render_template("draw_projection.html", data=projection_data, data_obj=dictionary_data, method='SpectralEmbedding')


@app.route('/projection/rfe')
def rfe():
    model = MyRandomForestEmbedding(source_data)
    projection_data = model.return_data
    projection_data = projection_data.tolist()
    return render_template("draw_projection.html", data=projection_data, data_obj=dictionary_data, method='RandomForestEmbedding')


@app.route('/cluster')
def cluster():
    return render_template('cluster.html')


@app.route('/cluster/kmeans')
def kmeans():
    n_clusters = 3
    if request.args.get('k_clusters'):
        n_clusters = int(request.args.get('k_clusters'))
    model = MyKmeans()
    clustering = model.fit(source_data, n_clusters)
    labels = clustering.labels_
    labels = labels.tolist()
    clusters = np.unique(labels).size

    pca = MyPCA(source_data)
    data_pca = pca.return_data
    samples, features = data_pca.shape
    data_pca = data_pca.tolist()
    for i in range(samples):
        data_pca[i].append(labels[i])
    return render_template("cluster.html", data=data_pca, data_obj=dictionary_data, clusters=clusters, method='KMeans')


@app.route('/cluster/minibatchkmeans')
def minibatchkmeans():
    n_clusters = 3
    if request.args.get('mk_clusters'):
        n_clusters = int(request.args.get('mk_clusters'))
    model = MyMiniBatchKMeans()
    clustering = model.fit(source_data, n_clusters)
    labels = clustering.labels_
    labels = labels.tolist()
    clusters = np.unique(labels).size

    pca = MyPCA(source_data)
    data_pca = pca.return_data
    samples, features = data_pca.shape
    data_pca = data_pca.tolist()
    for i in range(samples):
        data_pca[i].append(labels[i])
    return render_template("cluster.html", data=data_pca, data_obj=dictionary_data, clusters=clusters, method='MiniBatchKMeans')


@app.route('/cluster/meanshift')
def meanshift():
    global source_data
    bandwidth = None
    if request.args.get('bandwidth'):
        bandwidth = float(request.args.get('bandwidth'))
    model = MyMeanShift()
    clustering = model.fit(source_data, bandwidth)
    labels = clustering.labels_
    labels = labels.tolist()
    clusters = np.unique(labels).size

    pca = MyPCA(source_data)
    data_pca = pca.return_data
    samples, features = data_pca.shape
    data_pca = data_pca.tolist()
    for i in range(samples):
        data_pca[i].append(labels[i])
    return render_template("cluster.html", data=data_pca, data_obj=dictionary_data, clusters=clusters, method='MeanShift')


@app.route('/cluster/ap')
def ap():
    damping = 0.5
    preference = None
    if request.args.get('damping'):
        damping = float(request.args.get('damping'))
    if request.args.get('preference'):
        preference = float(request.args.get('preference'))
    model = MyAP()
    clustering = model.fit(source_data, damping, preference)
    labels = clustering.labels_
    labels = labels.tolist()
    clusters = np.unique(labels).size

    pca = MyPCA(source_data)
    data_pca = pca.return_data
    samples, features = data_pca.shape
    data_pca = data_pca.tolist()
    for i in range(samples):
        data_pca[i].append(labels[i])
    return render_template("cluster.html", data=data_pca, data_obj=dictionary_data, clusters=clusters, method='AffinityPropagation')


@app.route('/cluster/birch')
def birch():
    threshold = 0.5
    branching_factor = 50
    if request.args.get('threshold'):
        threshold = float(request.args.get('threshold'))
    if request.args.get('branching_factor'):
        branching_factor = int(request.args.get('branching_factor'))
    model = MyBirch()
    clustering = model.fit(source_data, threshold, branching_factor)
    labels = clustering.labels_
    labels = labels.tolist()
    clusters = np.unique(labels).size

    pca = MyPCA(source_data)
    data_pca = pca.return_data
    samples, features = data_pca.shape
    data_pca = data_pca.tolist()
    for i in range(samples):
        data_pca[i].append(labels[i])
    return render_template("cluster.html", data=data_pca, data_obj=dictionary_data, clusters=clusters, method='Birch')


@app.route('/cluster/dbscan')
def dbscan():
    eps = 0.5
    min_samples = 5
    if request.args.get('eps'):
        eps = float(request.args.get('eps'))
    if request.args.get('min_samples'):
        min_samples = int(request.args.get('min_samples'))
    model = MyDBSCAN()
    clustering = model.fit(source_data, eps, min_samples)
    labels = clustering.labels_
    labels = labels.tolist()
    clusters = np.unique(labels).size

    pca = MyPCA(source_data)
    data_pca = pca.return_data
    samples, features = data_pca.shape
    data_pca = data_pca.tolist()
    for i in range(samples):
        data_pca[i].append(labels[i])
    return render_template("cluster.html", data=data_pca, data_obj=dictionary_data, clusters=clusters, method='DBSCAN')


@app.route('/cluster/hdbscan')
def hdbscan():
    global source_data
    min_cluster_size = 0.5
    min_samples = 5
    alpha = 1.0
    cluster_selection_method = 'eom'
    if request.args.get('min_cluster_size'):
        min_cluster_size = int(request.args.get('min_cluster_size'))
    if request.args.get('min_samples'):
        min_samples = int(request.args.get('min_samples'))
    if request.args.get('alpha'):
        alpha = float(request.args.get('alpha'))
    if request.args.get('cluster_selection_method'):
        cluster_selection_method = int(request.args.get('cluster_selection_method'))
    model = MyHDBSCAN()
    clustering = model.fit(source_data, min_cluster_size, min_samples, alpha, cluster_selection_method)
    labels = clustering.labels_
    labels = labels.tolist()
    clusters = np.unique(labels).size

    pca = MyPCA(source_data)
    data_pca = pca.return_data
    samples, features = data_pca.shape
    data_pca = data_pca.tolist()
    for i in range(samples):
        data_pca[i].append(labels[i])
    return render_template("cluster.html", data=data_pca, data_obj=dictionary_data, clusters=clusters, method='HDBSCAN')


@app.route('/cluster/gmm')
def gmm():
    n_clusters = 1
    cov_type = 'full'
    if request.args.get('g_clusters'):
        n_clusters = int(request.args.get('g_clusters'))
    if request.args.get('cov_type'):
        cov_type = request.args.get('cov_type')
    model = MyGMM()
    clustering, labels = model.fit(source_data, n_clusters, cov_type)
    labels = labels.tolist()
    clusters = np.unique(labels).size

    pca = MyPCA(source_data)
    data_pca = pca.return_data
    samples, features = data_pca.shape
    data_pca = data_pca.tolist()
    for i in range(samples):
        data_pca[i].append(labels[i])
    return render_template("cluster.html", data=data_pca, data_obj=dictionary_data, clusters=clusters, method='GaussianMixture')


@app.route('/cluster/hierarchical')
def hierarchical():
    method = 'average'
    if request.args.get('linkage'):
        method = request.args.get('linkage')
    model = MyHierarchical()
    clustering, labels = model.fit(source_data, method)
    labels = labels.tolist()
    clusters = np.unique(labels).size

    pca = MyPCA(source_data)
    data_pca = pca.return_data
    samples, features = data_pca.shape
    data_pca = data_pca.tolist()
    for i in range(samples):
        data_pca[i].append(labels[i])
    return render_template("cluster.html", data=data_pca, data_obj=dictionary_data, clusters=clusters, method='hierarchy')


@app.route('/cluster/ac', methods=['POST', 'GET'])
def ac():
    n_clusters = 2
    if request.args.get('a_clusters'):
        n_clusters = int(request.args.get('a_clusters'))
    model = MyAC()
    clustering = model.fit(source_data, n_clusters)
    labels = clustering.labels_
    labels = labels.tolist()
    clusters = np.unique(labels).size

    pca = MyPCA(source_data)
    data_pca = pca.return_data
    samples, features = data_pca.shape
    data_pca = data_pca.tolist()
    for i in range(samples):
        data_pca[i].append(labels[i])
    return render_template("cluster.html", data=data_pca, data_obj=dictionary_data, clusters=clusters, method='AgglomerativeClustering')


@app.route('/cluster/sc', methods=['POST', 'GET'])
def sc():
    n_clusters = 8
    if request.args.get('s_clusters'):
        n_clusters = int(request.args.get('s_clusters'))
    model = MySC()
    clustering = model.fit(source_data, n_clusters)
    labels = clustering.labels_
    labels = labels.tolist()
    clusters = np.unique(labels).size

    pca = MyPCA(source_data)
    data_pca = pca.return_data
    samples, features = data_pca.shape
    data_pca = data_pca.tolist()
    for i in range(samples):
        data_pca[i].append(labels[i])
    return render_template("cluster.html", data=data_pca, data_obj=dictionary_data, clusters=clusters, method='SpectralClustering')


@app.route('/cluster/fa', methods=['POST', 'GET'])
def fa():
    n_clusters = 2
    method = 'ward'
    if request.args.get('f_clusters'):
        n_clusters = int(request.args.get('f_clusters'))
    if request.args.get('linkage'):
        method = request.args.get('linkage')
    model = MyFeatureAgglomeration()
    clustering = model.fit(source_data, n_clusters, method)
    labels = clustering.labels_
    labels = labels.tolist()
    clusters = np.unique(labels).size

    pca = MyPCA(source_data)
    data_pca = pca.return_data
    samples, features = data_pca.shape
    data_pca = data_pca.tolist()
    for i in range(samples):
        data_pca[i].append(labels[i])
    return render_template("cluster.html", data=data_pca, data_obj=dictionary_data, clusters=clusters, method='FeatureAgglomeration')


@app.route('/visualization')
def visualization():
    return render_template('visualization.html')


@app.route('/visualization/parallel_coordinates')
def parallel_coordinates():
    return render_template('parallel_coordinates.html',data=source_data, title=data_frame)


@app.route('/visualization/radviz')
def radviz():
    return render_template("radviz.html", data=dictionary_data, title=data_frame, data_obj= dictionary_data)


@app.route('/visualization/scatter', methods=['POST', 'GET'])
def scatter():
    return render_template("scatter_plot.html", data=dictionary_data, title=data_frame, list_scatter=dictionary_data_title)
