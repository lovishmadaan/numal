import numpy as np
# from makeNetwork import *
import matplotlib.pyplot as plt

def applyA(network, v):
    m, tuples = network
    Av = np.zeros((m, 1))
    Av[0] = v[0]
    Av[m - 1] = v[m - 1]
    for tup in tuples:
        i, j, r_ij = tup
        Av[i] += (v[i] - v[j]) / r_ij
        Av[j] += (v[j] - v[i]) / r_ij
    return Av

def getB(network):
    m, _ = network
    b = np.zeros((m, 1))
    b[0] = -1
    return b

def cg(Afun, b, tolerance):
    m = b.shape[0]
    x = np.zeros((m, 1))
    r = np.array(b)
    p = np.array(r)
    normB = np.linalg.norm(b)
    residuals = [np.linalg.norm(r) / normB]
    while True:
        if residuals[-1] < tolerance:
            break
        alpha = (r.T @ r) / (p.T @ Afun(p))
        x += alpha * p
        rprev = np.array(r)
        r -= alpha * Afun(p)
        residuals.append(np.linalg.norm(r) / normB)
        beta = (r.T @ r) / (rprev.T @ rprev)
        p = r + beta * p
    return x, residuals

def getDiag(network):
    m, tuples = network
    d = np.zeros((m, 1))
    d[0] = 1
    d[m - 1] = 1
    for tup in tuples:
        i, j, r_ij = tup
        d[i] += 1 / r_ij
        d[j] += 1 / r_ij
    return d

def pcg(Afun, b, d, tolerance):
    C = d ** 0.5
    Cinv = 1 / C
    bModified = Cinv * b
    x, residuals = cg(lambda v: Cinv * (Afun(Cinv * v)), bModified, tolerance)
    return Cinv * x, residuals


# network1 = makeNetwork('random1', 1000)
# x1, residuals1 = cg(lambda v: applyA(network1, v), -getB(network1), 1e-06)
# plt.plot(np.log(residuals1), '-o', markersize=3, label = 'cg')
# plt.rcParams["font.family"] = "serif"
# plt.rcParams["mathtext.fontset"] = "dejavuserif"
# plt.xlabel('n')
# plt.ylabel(r'$\log\frac{||r_n||_2}{||b||_2}$', rotation=0, labelpad=11)
# plt.legend()
# plt.savefig('fig1.png', dpi=500, bbox_inches='tight')
# plt.show()

# network2 = makeNetwork('random2', 1000)
# x1, residuals1 = cg(lambda v: applyA(network2, v), -getB(network2), 1e-06)
# x2, residuals2 = pcg(lambda v: applyA(network2, v), -getB(network2), getDiag(network2), 1e-06)
# plt.figure()
# plt.plot(np.log(residuals1), '-o', markersize=2, label = 'cg')
# plt.plot(np.log(residuals2), '-o', markersize=2, label = 'pcg')
# plt.rcParams["font.family"] = "serif"
# plt.rcParams["mathtext.fontset"] = "dejavuserif"
# plt.xlabel('n')
# plt.ylabel(r'$\log\frac{||r_n||_2}{||b||_2}$', rotation=0, labelpad=11)
# plt.legend()
# plt.savefig('fig2.png', dpi=500, bbox_inches='tight')
# plt.show()