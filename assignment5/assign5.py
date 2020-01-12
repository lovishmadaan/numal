from hw5starter import *

def gradientDescentStep(f, g, x):
    alpha, beta, step = 0.3, 0.5, 1

    func = f(x)
    grad_func = g(x)
    grad_x = -g(x)

    while(f(x + step * grad_x) > f(x) + alpha * step * np.dot(grad_func, grad_x)):
        step *= beta

    return x + step * grad_x

def f(y):
    x = np.linspace(-1.5, 1.5, 13)

    func = (y[0] ** 2) + (y[10] ** 2)
    for i in range(1, 11):
        func += (y[i] - y[i - 1]) ** 2
    func = 0.5 * 10 * func

    for i in range(11):
        func += 0.1 * (fObs(np.array([x[i + 1], y[i]])))

    return func

def g(y):
    x = np.linspace(-1.5, 1.5, 13)

    grad = np.zeros(11)

    for i in range(11):
        if i == 0:
            yprev = 0
        else:
            yprev = y[i - 1]
        if i == 10:
            ynext = 0
        else:
            ynext = y[i + 1]
        grad[i] = 10 * (-yprev + 2 * y[i] - ynext) + 0.1 * gObs(np.array([x[i + 1], y[i]]))[1]
    return grad

def Htilde():
    matrix = np.diag(20 * np.ones(11), k=0) + np.diag(-10 * np.ones(10), k=-1) + np.diag(-10 * np.ones(10), k=1)
    return matrix

def steepestDescentStep(f, g, P, x):
    alpha, beta, step = 0.3, 0.5, 1

    func = f(x)
    grad_func = g(x)
    grad_x = np.linalg.solve(P, -g(x))

    while(f(x + step * grad_x) > f(x) + alpha * step * np.dot(grad_func.T, grad_x)):
        step *= beta

    return x + step * grad_x

if __name__ == "__main__":
    # Part a
    # print(g(np.zeros(11)))
    # x = np.zeros((20, 2))
    # for i in range(1, 20):
    #     x[i] = gradientDescentStep(fObs, gObs, x[i - 1])
    
    # visualizeObs()
    # c = np.linspace(0, 1, 20)
    # plt.scatter(x[:, 0], x[:, 1], c=c, cmap='hot')
    # print(x)
    # plt.colorbar()
    # plt.savefig('fig1.png', bbox_inches='tight', dpi=500)
    # plt.show()

    # Part b
    # iters = 20
    # y = np.zeros((iters + 1, 11))
    # gy_norm = np.zeros((iters + 1))
    # gy_norm[0] = np.linalg.norm(g(y[0]))
    # for i in range(iters):
    #     y[i + 1] = gradientDescentStep(f, g, y[i])
    #     gy_norm[i + 1] = np.linalg.norm(g(y[i + 1]))
    
    # visualizeTrajectory(y[iters], g(y[iters]))
    # plt.savefig('fig2.png', bbox_inches='tight', dpi=500)
    # plt.show()

    # plt.plot(np.log(gy_norm), '-o', label=r'$||g(y^{(k))}$||')
    # plt.title('Gradient Descent')
    # plt.xlabel('Iteration Number')
    # plt.ylabel(r'log($||g(y^{(k))}$||)')
    # plt.legend()
    # plt.savefig('fig3.png', bbox_inches='tight', dpi=500)
    # plt.show()

    print(f(np.zeros(11)))
    print(g(np.zeros(11)))
    exit()

    # Part c
    iters = 20
    y = np.zeros((iters + 1, 11))
    gy_norm = np.zeros((iters + 1))
    gy_norm[0] = np.linalg.norm(g(y[0]))
    for i in range(iters):
        y[i + 1] = steepestDescentStep(f, g, Htilde(), y[i])
        gy_norm[i + 1] = np.linalg.norm(g(y[i + 1]))
        print(y[i + 1], gy_norm[i + 1])
    
    visualizeTrajectory(y[iters], g(y[iters]))
    plt.savefig('fig4.png', bbox_inches='tight', dpi=500)
    plt.show()

    plt.plot(np.log(gy_norm), '-o', label=r'$||g(y^{(k))}$||')
    plt.title('Steepest Descent in Quadratic Norm')
    plt.xlabel('Iteration Number')
    plt.ylabel(r'log($||g(y^{(k))}$||)')
    plt.legend()
    plt.savefig('fig5.png', bbox_inches='tight', dpi=500)
    plt.show()