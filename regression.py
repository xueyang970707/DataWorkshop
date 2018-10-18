# Simple Regession
import numpy as np
from sklearn.linear_model import LinearRegression

'''
x = [1,2,3,4,5,6,7,8,9]
y = [6,5,7,10,11,13,12,14,15]'''


def predict(b0, b1, x):
    return b0 + b1 * x


def fitSLR(data):
    x = []
    y = []
    for i in range(len(data)):
        x.append(data[i][0])
        y.append(data[i][1])
    n = len(x)
    denominator = 0
    numerator = 0
    for i in range(0, n):
        numerator += (x[i] - np.mean(x) * (y[i] - np.mean(y)))
        denominator += (x[i] - np.mean(x)) ** 2

    b1 = numerator / float(denominator)
    #     b0 = np.mean(y)/float(np.mean(x))
    b0 = np.mean(y) - b1 * np.mean(x)

    x1 = [min(x), max(x)]
    y1 = [predict(b0, b1, min(x)), predict(b0, b1, max(x))]
    result = []
    result.append(x1)
    result.append(y1)
    return x1, y1

'''
plt.scatter(x, y)

b0, b1 = fitSLR(x, y)

x_test = 19
print(predict(b0, b1, x_test))

x1 = [min(x), max(x)]
y1 = [predict(b0, b1, min(x)), predict(b0, b1, max(x))]
plt.plot(x1, y1)
plt.show()'''
