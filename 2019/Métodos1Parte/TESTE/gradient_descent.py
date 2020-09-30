import neural_network
from neural_network import backpropagation
from neural_network import feedforward
import numpy as np
from numpy import linalg as la 
import matplotlib.pyplot as plt
import settings
from settings import w1
from settings import w2




#erro quadrático (funcao a ser minimizada)
def eq(x,y):
    return (feedforward(x[0]) - y[0])**2 +(feedforward(x[1]) - y[1])**2 
    +(feedforward(x[2]) - y[2])**2 +(feedforward(x[3]) - y[3])**2
    +(feedforward(x[4]) - y[4])**2


#funcao que calcula o gradiente da funcao de erro quadratico
def grad_f(x,y):
  temp_grad = 0
  temp_grad_h = 0
  for i in range(len(x)):
    temp_grad, temp_grad_h =  backpropagation(x[i],y[i])
    if(i==0):
      grad = temp_grad
      grad_h = temp_grad_h
    else:
      grad = grad + temp_grad
      grad_h = grad_h + temp_grad_h

  return grad, grad_h



def metodo_gradiente(X, y, t):
    global w1
    global w2
    n_iter = 0
    while True:
    	
        grad, grad_h = grad_f(X, y)
        grad = np.transpose(grad)
        grad_h = np.transpose(grad_h)

        w1 = w1 - t*grad_h
        w2 = w2 - t*grad
        log_norm_gd = np.log10(la.norm(np.matrix([[grad_h[0,0]],[grad_h[0,1]],[grad[0,0]],[grad[1,0]]])))
        #print('w1: ',w1)
        #print('w2: ',w2)
        #print(log_norm_gd)
        if(log_norm_gd < -6.0):
          break
        n_iter = n_iter + 1

    m = (w1[0,0]*w2[0,0] + w1[1,0]*w2[0,1])
    print('m: ',  m)

    ex = np.linspace(0,6,10)
    ey = m*ex

    plt.axis([0,6,0,30])
    plt.scatter(X,[1,4,9,16,25],s = 30, c = "red")

    plt.plot(ex,ey)
    plt.show()
