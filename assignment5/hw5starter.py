import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt

def fObs(p):
    """Total penalty function at point p from all obstacles"""
    f = 0
    for obs in obstacles(p):
        f += fPenalty(obs[0])
    return f

def gObs(p):
    """Gradient of penalty function at point p"""
    g = np.array([0.,0.])
    for obs in obstacles(p):
        g += gPenalty(obs[0])*obs[1]
    return g

def visualizeObs():
    """"Visualize the obstacle field fObs. Call plt.show() to display"""
    fcontourf(fObs, [-2, 2], [-1, 1], [0, 10])

def visualizeTrajectory(y, g):
    """"Visualize a trajectory, and its negative gradient as arrows"""
    visualizeObs()
    x = np.linspace(-1.5, 1.5, 13)[1:-1]
    plt.plot(np.concatenate(([-1.5],x,[1.5])), np.concatenate(([0],y,[0])), color='black', marker='+')
    if g is not None:
        for i in range(y.size):
            plt.arrow(x[i], y[i], 0, -0.5*g[i], color='blue', head_width=0.05)

#### Helper functions below, can be ignored ####

def fPenalty(d):
    """Penalty function for being at signed distance d from obstacle"""
    return 1/(max(d,-0.1)+0.2)

def gPenalty(d):
    """Derivative of penalty function at signed distance d"""
    return -1/(d+0.2)**2 if d > -0.1 else 0

def obstacles(p):
    """Signed distance to each obstacle and its gradient"""
    c1 = np.array([-0.5,-1.])
    r1 = 1.
    c2 = np.array([0.75,0.5])
    r2 = 0.5
    return [
        (p[0] + 2, np.array([1.,0.])), # left
        (2 - p[0], np.array([-1.,0.])), # right
        (p[1] + 1, np.array([0.,1.])), # bottom
        (1 - p[1], np.array([0.,-1.])), # top
        (norm(p - c1) - r1, (p - c1)/norm(p - c1)), # circle 1
        (norm(p - c2) - r2, (p - c2)/norm(p - c2)) # circle 2
    ]

def fcontourf(f, x1range, x2range, yrange, **kwargs):
    """Draw filled contours of the function f over the specified range."""
    x1s = np.linspace(x1range[0], x1range[1])
    x2s = np.linspace(x2range[0], x2range[1])
    ys = np.linspace(yrange[0], yrange[1], 20)
    fs = [[f(np.array([x1,x2])) for x1 in x1s] for x2 in x2s]
    plt.contourf(x1s, x2s, fs, ys, **kwargs)
    plt.axis('scaled')
