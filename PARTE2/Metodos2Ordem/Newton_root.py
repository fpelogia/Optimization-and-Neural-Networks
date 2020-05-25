import numpy as np
import numpy.linalg as la
import time

def func(x):
    return np.array([x[1] - x[0]**2, x[0]**2 + x[1]**2 - 1])

def jacf(x):
    return np.array([
                np.array([-2*x[0], 1]),
                np.array([2*x[0],2*x[1]])
    ])

x = (2,3)

def Newton(x):
    n_iter = 0
    while la.norm(func(x)) > 1e-5:
        d = la.solve(jacf(x), -1*func(x))
        x = x + d
        n_iter = n_iter + 1
    return x, n_iter


start_time = time.time()
res, it = Newton(x)
end_time = time.time()
print('x:', res)
print('f(x):', func(res))
print('n_iter:', it)
print(f'Tempo de execução: {end_time - start_time}s')
        