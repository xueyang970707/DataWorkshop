import numpy as np
import pandas as pd
from scipy.stats import mode


class Statistic:
    mean = []
    median = []
    mode = None
    min = []
    max = []
    var = []
    cov = []
    sampling_data = []

    def __init__(self, source, frame):
        source = pd.DataFrame.from_records(source, columns=frame)
        mean = np.mean(source, axis=0)
        self.mean = []
        for m in mean:
            m = m * 100
            m = round(m)
            m = m / 100
            self.mean.append(m)
        self.median = np.median(source, axis=0)
        self.mode = mode(source, axis=0)
        self.min = np.min(source, axis=0)
        self.max = np.max(source, axis=0)
        var = np.var(source, axis=0)
        self.var = []
        for m in var:
            m = m * 100
            m = round(m)
            m = m / 100
            self.var.append(m)
        self.cov = np.cov(source)

