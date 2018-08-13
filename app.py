import ast
import numpy as np
from flask import Flask, request, json, redirect, render_template, url_for, flash
from statistics import Statistics
from projection import MyGaussianRandomProjection, MySparseRandomProjection, \
    MyPCA, MyMiniBatchSparsePCA, MyKernelPCA, MySparsePCA, MyRandomizedPCA, MyIncrementalPCA, \
     MyNMF, MyLDA, MyFastICA, MyMDS, MyISOMAP, MyLLE, MyHLLE, MyMLLE, MyLTSA, MyTSNE, MySpectralEmbedding, MyRandomForestEmbedding
from cluster import MyAP, MyBirch, MyDBSCAN, MyGMM, MyHierarchical, MyKmeans, MyMeanShift, MySC, MyAC,\
    MyMiniBatchKMeans, MyHDBSCAN, MyFeatureAgglomeration


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


@app.route('/data_workshop', methods=['GET', 'POST'])
def data_workshop():
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
    return render_template('index.html')


@app.route('/statistics')
def statistic():
    stt = Statistics(source_data, data_frame)
    mode = stt.mode
    return render_template('draw_statistics.html', mean=stt.mean, median=stt.median, mode=mode, min=stt.min, max=stt.max,
                           var=stt.var, cov=stt.cov, frame=data_frame)


@app.route('/cluster_projection')
def cluster_projection():
    n_clusters = 3
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
    return render_template('draw_cluster_projection.html', data=data_pca, data_obj=dictionary_data, clusters=clusters, method='KMeans')


@app.route('/visualization', methods=['POST', 'GET'])
def upload_visualization():
    return render_template("draw_visualization.html", data=dictionary_data, title=data_frame, data_obj= dictionary_data, method='Radviz')

