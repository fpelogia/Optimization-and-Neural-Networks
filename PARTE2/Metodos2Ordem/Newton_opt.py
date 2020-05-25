import numpy as np
import numpy.linalg as la
import time

def func(x):
    return np.exp(x[0] + x[1]) + x[0]**2 + x[1]**2


def gradf(x):
    return np.array([
                np.exp(x[0] + x[1]) + 2*x[0],
                np.exp(x[0] + x[1]) + 2*x[1]
    ])
def hessf(x):
    return np.array([
        np.array([np.exp(x[0] + x[1]) + 2, np.exp(x[0] + x[1])]),
        np.array([np.exp(x[0] + x[1]) ,np.exp(x[0] + x[1]) + 2])
        ])

x = (1,2)

def Newton(x):
    n_iter = 0
    while la.norm(gradf(x)) > 1e-5:
        d = la.solve(hessf(x), -1*gradf(x))
        x = x + d
        n_iter = n_iter + 1
    return x, n_iter

def Newton_ch(x):
    n_iter = 0
    while la.norm(gradf(x)) > 1e-5:

        L = la.cholesky(hessf(x))
        v = la.solve(L, -1*gradf(x))
        d = la.solve(np.transpose(L), v)

        x = x + d
        n_iter = n_iter + 1
    return x, n_iter


       



print('\n Método de Newton calculando o sistema')
start_time = time.time()
res, it = Newton(x)
end_time = time.time()
print('x:', res)
print('f(x):', func(res))
print('n_iter:', it)
print(f'Tempo de execução: {end_time - start_time}s')

print('\n Método de Newton utilizando cholesky')
start_time = time.time()
res, it = Newton_ch(x)
end_time = time.time()
print('x:', res)
print('f(x):', func(res))
print('n_iter:', it)
print(f'Tempo de execução: {end_time - start_time}s')
        