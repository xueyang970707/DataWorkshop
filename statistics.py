import numpy as np
import pandas as pd
from scipy.stats import mode


def tran2float(values_array):
    result_array = []
    for m in values_array:
        m = m * 100
        m = round(m)
        m = m / 100
        result_array.append(m)
    return result_array


class Statistics:
    mean = []
    median = []
    mode = []
    min = []
    max = []
    var = []
    cov = []
    corr = []
    sampling_data = []

    def __init__(self, source, frame):
        source = pd.DataFrame.from_records(source, columns=frame)
        self.mean = tran2float(np.mean(source, axis=0))
        self.median = tran2float(np.median(source, axis=0))
        self.mode = tran2float(mode(source, axis=0).mode[0])
        self.min = tran2float(np.min(source, axis=0))
        self.max = tran2float(np.max(source, axis=0))
        self.var = tran2float(np.var(source, axis=0))
        self.cov = np.cov(source)
        self.corr = np.array(source.corr(method='spearman')).tolist()