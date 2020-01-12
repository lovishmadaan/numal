import numpy as np
from fcontour import *
import matplotlib.cm as cm

# def p(x):
#     return x**3 - x**2 - x + 1

# def pdash(x):
#     return 3 * (x**2) - 2 * x - 1

# def newton(x0):
#     x = x0
#     for k in range(4):
#         x = x - (p(x) / pdash(x))
#         print(x)

# print('x0 = -2')
# newton(-2)
# print('x0 = 2')
# newton(2)

class Conic:

    def __init__(self, a, b, c, p, q, r):
        self.a = a
        self.b = b
        self.c = c
        self.p = p
        self.q = q
        self.r = r
    
    def evaluate(self, x):
        return self.a * (x[0] ** 2) + self.b * x[0] * x[1] + self.c * (x[1] ** 2) + self.p * x[0] + self.q * x[1] + self.r
    
    def jacobian(self, x):
        derivative = np.zeros(2)
        derivative[0] = 2 * self.a * x[0] + self.b * x[1] + self.p
        derivative[1] = self.b * x[0] + 2 * self.c * x[1] + self.q
        return derivative
    
    def linearize(self, x):
        x = np.array(x)
        def ftilde(y):
            y = np.array(y)
            return self.evaluate(x) + np.dot(self.jacobian(x), y - x)
        return ftilde

def newtonStep(conic1, conic2, x):
    x = np.array(x)
    f = np.array([conic1.evaluate(x), conic2.evaluate(x)])
    J = np.zeros((2, 2))
    J[0, :] = conic1.jacobian(x)
    J[1, :] = conic2.jacobian(x)
    s = np.linalg.solve(J, -f)
    return x + s

# conic1 = Conic(1, 0, 4, 0, 0, -16)
# conic2 = Conic(4, 0, 1, 0, 0, -16)


# fcontour(conic1.evaluate, [-5, 5], [-5, 5], colors='r')
# fcontour(conic2.evaluate, [-5, 5], [-5, 5], colors='b')
# plt.show()

# exit()

# conic1 = Conic(1, 0, 4, 0, 0, -16)
# conic2 = Conic(4, 0, 1, 0, 0, -16)

# colors = ['blue', 'magenta', 'cyan', 'green']
# x = np.array([6.0,7.0])
# for i in range(5):
#     plt.figure()
#     p1, = plt.plot(x[0], x[1], marker='*', markersize=10, linestyle='')
#     fcontour(conic1.evaluate, [-5, 7], [-5, 7], colors=colors[0])
#     fcontour(conic2.evaluate, [-5, 7], [-5, 7], colors=colors[1])
#     fcontour(conic1.linearize(x),[-5, 7], [-5, 7], colors=colors[2])
#     fcontour(conic2.linearize(x),[-5, 7], [-5, 7], colors=colors[3])
#     x = newtonStep(conic1, conic2, x)
#     p2, = plt.plot(x[0], x[1], marker='o', markersize=5, linestyle='')
#     proxy = [plt.Rectangle((0, 0), 0.5, 0.5, fc = color) for color in colors]
#     plt.legend([p1, p2] + proxy, [r'$x^{(' + str(i) + ')}$', r'$x^{(' + str(i + 1) + ')}$', r'$f_{1}(x) = 0$', r'$f_{2}(x) = 0$', r'$\tilde{f_{1}}(x) = 0$', r'$\tilde{f_{2}}(x) = 0$'])
#     plt.xlabel(r'$x_1$')
#     plt.ylabel(r'$x_2$')
#     plt.savefig('fig' + str(i) + '.png', bbox_inches='tight', dpi=500)
#     print(x)