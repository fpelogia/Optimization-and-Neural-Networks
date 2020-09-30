import numpy as np
import settings
from settings import w1
from settings import w2

func = lambda x: max(x,0)
func = np.vectorize(func)

def d_func(x):
    dx = x
    dx[x<0.0] = 0
    dx[x>=0.0] = 1
    return dx



def backpropagation(I, y):
    H_unac = np.dot(w1,I) # unac indica que é a versao não ativada da camada
    H = func(H_unac)
    O_unac = np.dot(w2,H)
    O = func(O_unac)

    print('w1: ',w1)
    print('w2: ',w2)

    error = O - y
    alpha = error*d_func(O_unac)
    
    
    grad = np.transpose(np.multiply(alpha , H)) 
    
    erro_h = np.dot(np.transpose(w2), alpha)
    
    alpha_h = np.multiply(erro_h, d_func(H_unac))
    
    grad_h = np.multiply(alpha_h, np.transpose(np.tile(I, (1, 2))))

    return grad, grad_h


def feedforward(I):
    H_unac = np.dot(w1,I) # unac indica que é a versao não ativada da camada
    H = func(H_unac)
    O_unac = np.dot(w2,H)
    O = func(O_unac)
    return O
