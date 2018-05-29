import numpy as np
import matplotlib.pyplot as plt


class Approximator:
    def __init__(self, N, M, L, mu, target_func, nonLinear):
        # Training data length
        self.N = N
        # Number of neurons
        self.M = M
        # Number of update loops
        self.L = L
        # Learning rate
        self.mu = mu
        # Non-Linear function
        self.nonLinear = nonLinear
        # define training input data (points in unit line)
        self.x_data = np.linspace(0, 1, self.N)
        # Make data matrix to deal with bias
        self.X = np.array([self.x_data, np.ones(self.N)])
        # generate training data from given function f
        self.t = target_func(self.x_data)
        # initialize weights (Gaussian initialization)
        self.W = 0.1*np.random.randn(self.M,2)
        self.alpha = 0.1*np.random.randn(self.M,1)
        # to trace MSE cost function
        self.J_MSE = np.empty(self.L)

    def sigma(self, x):
        return 1/(1+np.exp(-x))

    def d_sigma(self, x):
        return self.sigma(x) * (1 - self.sigma(x))

    def relu(self, x):
        return np.maximum(0, x)

    def d_relu(self, x):
        return (x > 0).astype(int)

    def tanh(self, x):
        return (np.exp(x)-np.exp(-x)) / (np.exp(x)+np.exp(-x))

    def d_tanh(self, x):
        return 1/(1+np.power(x, 2))

    # Weight update
    def W_update(self):
        actF = eval('self.'+ self.nonLinear)
        d_actF = eval('self.'+'d_'+ self.nonLinear)
        for epoch in range(self.L):
            a = np.matmul(self.W, self.X)
            F = np.matmul(self.alpha.T, actF(a))
            d_alpha = 2*np.matmul(self.sigma(a), (F - self.t).T)/self.N
            d_W0 = 2*np.multiply(self.alpha, np.matmul(d_actF(a), np.multiply(self.x_data, (F - self.t)).T))
            d_W1 = 2*np.multiply(self.alpha, np.matmul(d_actF(a), (F - self.t).T))
            self.alpha = self.alpha-self.mu*d_alpha
            self.W = self.W-self.mu*np.concatenate((d_W0, d_W1), axis=1)
            self.J_MSE[epoch]= np.linalg.norm(F-self.t)**2
        return self.J_MSE


def simulate(N, M, L, mu, target_func, nonLinear):
    '''
    Setting (N, M, L, mu) = (Number of data, Number of neuron, Iteration number, Learning rate),
    approximating target_func using Universal Approximation Theorem. Show loss and approximation result
    '''
    _obj = Approximator(N=N, M=M, L=L, mu=mu,target_func=target_func, nonLinear=nonLinear)
    loss = _obj.W_update()
    plt.figure(1)
    plt.plot(loss)
    plt.savefig('L(N, M, L, mu, Non Linear)='+str(N)+','+str(M)+','+str(L)+','+str(mu)+','+nonLinear+'.png')

    act = eval('_obj.' + nonLinear)
    x_eval=np.linspace(0,1,2*N)
    f_eval=f_test(x_eval)
    X_eval=np.array([x_eval, np.ones(2*N)])
    F_eval=np.matmul(_obj.alpha.T, act(np.matmul(_obj.W,X_eval)))
    plt.figure(2)
    plt.plot(x_eval, f_eval)
    plt.plot(x_eval, F_eval.T)
    plt.savefig('A(N, M, L, mu, Non Linear)='+str(N)+','+str(M)+','+str(L)+','+str(mu)+','+nonLinear+'.png')
    plt.show()

## (1) cos(2 * pi * x) approximation

# define target function
def f_test(x):
    return np.cos(2*np.pi*x)

# With Logistic function
simulate(N=100, M=20, L=1000, mu=0.01, target_func=f_test, nonLinear='sigma')
simulate(N=100, M=30, L=10000, mu=0.01, target_func=f_test, nonLinear='sigma')

# With Relu
simulate(N=100, M=20, L=1000, mu=0.05, target_func=f_test, nonLinear='relu')
simulate(N=100, M=30, L=10000, mu=0.01, target_func=f_test, nonLinear='relu')

# With tanh
simulate(N=1000, M=20, L=5000, mu=0.01, target_func=f_test, nonLinear='tanh')
simulate(N=100, M=30, L=10000, mu=0.01, target_func=f_test, nonLinear='tanh')

## (2) sin(x) approximation

# define target function
def f_test(x):
    return np.sin(2*np.pi*x)

# With Logistic function
simulate(N=100, M=20, L=1000, mu=0.05, target_func=f_test, nonLinear='sigma')
simulate(N=1000, M=30, L=10000, mu=0.01, target_func=f_test, nonLinear='sigma')

# With Relu
simulate(N=100, M=20, L=1000, mu=0.05, target_func=f_test, nonLinear='relu')
simulate(N=1000, M=50, L=10000, mu=0.002, target_func=f_test, nonLinear='relu')

# With tanh
simulate(N=100, M=20, L=1000, mu=0.05, target_func=f_test, nonLinear='tanh')
simulate(N=100, M=30, L=10000, mu=0.01, target_func=f_test, nonLinear='tanh')
