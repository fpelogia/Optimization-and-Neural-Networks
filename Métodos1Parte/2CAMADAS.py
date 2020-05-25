import numpy as np
from numpy import linalg as la 
import matplotlib.pyplot as plt
import sys



X = [np.transpose(np.matrix([1])),
     np.transpose(np.matrix([2])),
     np.transpose(np.matrix([3])),
     np.transpose(np.matrix([4])),
     np.transpose(np.matrix([5]))
     ]


w1 = np.matrix([[1],[2]])
#w1 = np.matrix(np.random.normal(0.0, pow(2, -0.5),(2,1)))
w2 = np.matrix([[3,4],[5,6]])
w3 = np.matrix([7,8])
#w2 = np.matrix(np.random.normal(0.0, pow(1, -0.5),(1,2)))
y = [1,4,9,16,25]

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
    H2_unac = np.dot(w2,H)
    H2 = func(H2_unac)
    O_unac = np.dot(w3,H2)
    O = func(O_unac)

    error = O - y
    alpha = error*d_func(O_unac)
    
    
    grad = np.transpose(np.multiply(alpha , H)) 
    
    erro_h = np.dot(np.transpose(w3), alpha)
    
    alpha_h = np.multiply(erro_h, d_func(H2_unac))
    
    grad_h = np.multiply(alpha_h, np.transpose(np.tile(H, (1, 2))))

    erro_h2 = np.dot(np.transpose(w2), alpha_h)

    alpha_h2 = np.multiply(erro_h2, d_func(H_unac))

    grad_h2 = np.multiply(alpha_h2, np.transpose(np.tile(I, (1, 2)))) 

    return grad, grad_h, grad_h2


def feedforward(I):
    H_unac = np.dot(w1,I) # unac indica que é a versao não ativada da camada
    H = func(H_unac)
    H2_unac = np.dot(w2,H)
    H2 = func(H2_unac)
    O_unac = np.dot(w3,H2)
    O = func(O_unac)

    return O





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
    temp_grad, temp_grad_h, temp_grad_h2 =  backpropagation(x[i],y[i])
    if(i==0):
      grad = temp_grad
      grad_h = temp_grad_h
      grad_h2 = temp_grad_h2
    else:
      grad = grad + temp_grad
      grad_h = grad_h + temp_grad_h
      grad_h2 = grad_h2 + temp_grad_h2

  return grad, grad_h, grad_h2



def metodo_gradiente(X, y, t):
    global w1
    global w2
    n_iter = 0
    while True:
        
        grad, grad_h, grad_h2 = grad_f(X, y)
        grad = np.transpose(grad)
        grad_h = np.transpose(grad_h)
        grad_h2 = np.transpose(grad_h2)

        w1 = w1 - np.multiply(t, grad)
        w2 = w2 - np.multiply(t, grad_h)
        w3 = w3 - np.multiply(t, grad_h2)
        log_norm_gd = np.log10(la.norm(np.matrix([[grad_h[0,0]],[grad_h[0,1]],[grad[0,0]],[grad[1,0]]])))

        if(log_norm_gd < -6.0):
          break
        n_iter = n_iter + 1

    m = (w1[0,0]*w2[0,0] + w1[1,0]*w2[0,1])
    print('m: ',  m)
    print('n_iter', n_iter)

    ex = np.linspace(0,6,10)
    ey = m*ex

    plt.axis([0,6,0,30])
    plt.scatter(X,[1,4,9,16,25],s = 30, c = "red")

    plt.plot(ex,ey)
    plt.show()



metodo_gradiente(X, y, float(sys.argv[1]))