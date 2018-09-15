import numpy as np
import pandas as pd
import random


def simple_random_sampling(data,sample_number):
    idx_test = data.sample(n=sample_number)
    print(idx_test)
    return idx_test


def repetition_random_sampling(data,sample_number):
    for i in range(sample_number):
        dt=data.sample(n=1)
        if i == 0:
            df = dt
        else:
            df = pd.concat([df,dt])
    print(df)
    return df


def systematic_sampling(data, sampling_interval):
    a = data.shape[0]
    k = sampling_interval
    b = int(a/k)
    choosing = list()
    for i in range(k):
        choosing.append(i)
    choosing_ = random.sample(choosing,1)
    for i in choosing_:
        kth = i
    for i in range(b):
        if i == 0:
            system_sample = data[kth:kth+1]
        else:
            system_sample = pd.concat([system_sample, data[kth+i*k:kth+i*k+1]])
    return system_sample







