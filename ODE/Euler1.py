import numpy as np
import matplotlib.pyplot as plt


def c(x):
    return 1

def f(x):
    return np.sin(x)+np.cos(x)

def Euler(tspan, N, u0):
    h = tspan / N  # time space
    x = np.linspace(0, tspan, N+1)
    A = np.zeros((N, N))
    F = np.zeros((N, 1))
    u = np.zeros((N+1, 1))
    u[0, 0] = u0
    for i in range(N):
        A[i, i] = 1
        if i>0:
            A[i, i-1] = h*c(x[i])-1
        F[i,0] = (h*f(x[i])+u0)/(c(x[i])+2) if i==0 else h*f(x[i])
    u[1:] = np.linalg.inv(A).dot(F)
    return u


plt.figure(figsize=(10,3))
for i, N in enumerate([10, 50, 100]):
    x = np.linspace(0, 2*np.pi, N+1)
    ax = plt.subplot(1, 3, i+1)
    ax.plot(x, Euler(2*np.pi, N, 0), label='euler')
    ax.plot(x, np.sin(x), label='true')
    ax.legend(fontsize=16)
    ax.set_xlabel('t', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    ax.set_title(f'N = {N}', fontsize=16)
plt.show()