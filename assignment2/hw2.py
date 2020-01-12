import numpy as np

def split(img, n):
    (h, w, d) = img.shape
    newH = h // n
    newW = w // n
    newImg = np.empty((d * n * n, newH * newW))
    for i in range(newH):
        for j in range(newW):
            block = img[i * n : (1 + i) * n, j * n : (1 + j) * n, :]
            newImg[:, i * newW + j] = block.flatten()
    return newImg

def join(C, n, w, h):
    newH = h // n
    newW = w // n
    oldImg = np.empty((h, w, 3))
    for i in range(newH):
        for j in range(newW):
            layer = C[:, i * newW + j]
            oldImg[i * n : (1 + i) * n, j * n : (1 + j) * n, :] = layer.reshape(n, n, 3)
    return oldImg

def compress(C, r):
    (p, q) = C.shape
    (u, sig, vT) = np.linalg.svd(C)
    sigKeep = np.reshape(sig[ : r], (1, r))
    return (u[:, : r] * sigKeep, vT[ : r, :], np.linalg.norm(sig[r : ]) / np.linalg.norm(sig))

def relError(img, img2):
    return np.linalg.norm(img - img2) / np.linalg.norm(img)