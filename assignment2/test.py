from hw2 import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img = mpimg.imread('kodim04.png')
plt.imshow(img)
C = split(img, 16)
A, B, e_rel = compress(C, 50)
print(e_rel)
img2 = join(A @ B, 16, img.shape[1], img.shape[0])
print(relError(img, img2))
plt.figure()
plt.imshow(img2)
plt.show()