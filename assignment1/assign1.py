import numpy as np

def var1(X):
    n = len(X)
    valSquares = 0.0
    valSum = 0.0
    for val in X:
        valSquares += val ** 2
        valSum += val
    return (valSquares / n) - (valSum / n) ** 2

def var2(X):
    n = len(X)
    valMean = 0.0
    for val in X:
        valMean += val
    valMean /= n
    valVar = 0.0
    for val in X:
        valVar += (val - valMean) ** 2
    return valVar / n

def data1():
    a = 0.001 * np.array(list(range(1, 11)))
    a = 1e07 + a
    return a

def kahaan_sum(X):
    n = len(X)
    valSum = X[0];
    val = 0.0;
    for i in range(1, n):
        temp = X[i] - val
        t = valSum + temp
        val = (t - valSum) - temp
        valSum = t
    return valSum

def var3(X):
    n = len(X)
    valMean = kahaan_sum(X)
    valMean /= n
    valVar = kahaan_sum(np.multiply((X - valMean), (X - valMean)))
    return valVar / n

def data2():
    return np.array([1e-10]*10)
