import numpy as np
from numpy import linalg as la 
import matplotlib.pyplot as plt
import sys
import random


func = lambda x: max(x,0.0)
func = np.vectorize(func)

def d_func(x):
    dx = x
    dx[x<0.0] = 0
    dx[x>=0.0] = 1
    return dx



def backpropagation(I, y, w1, w2):
    H_unac = np.dot(w1,I) 
    H = func(H_unac)
    O_unac = np.dot(w2,H)
    O = func(O_unac)

    error = O - y
    alpha = error*d_func(O_unac)
    
    
    grad = np.transpose(np.multiply(alpha , H)) 
    
    erro_h = np.dot(np.transpose(w2), alpha)
    
    alpha_h = np.multiply(erro_h, d_func(H_unac))
    
    grad_h = np.multiply(alpha_h, np.transpose(np.tile(I, (1, 2))))

    return grad, grad_h


def feedforward(I,w1,w2):
    H_unac = np.dot(w1,I) # unac indica que é a versao não ativada da camada
    H = func(H_unac)
    O_unac = np.dot(w2,H)
    O = func(O_unac)
    return O





#erro quadrático (funcao a ser minimizada)
def eq(x,y,w1,w2):
    parte_1 = (feedforward(x[0],w1,w2) - y[0])**2
    parte_2 = (feedforward(x[1],w1,w2) - y[1])**2
    parte_3 = (feedforward(x[2],w1,w2) - y[2])**2
    parte_4 = (feedforward(x[3],w1,w2) - y[3])**2
    parte_5 = (feedforward(x[4],w1,w2) - y[4])**2

    return (parte_1 + parte_2 + parte_3 + parte_4 + parte_5)/5


#funcao que calcula o gradiente da funcao de erro quadratico
def grad_f(x,y,w1,w2):

    temp_grad = 0
    temp_grad_h = 0
    indices = random.sample(range(0, len(x)), 5)
    for i in range(5):
        temp_grad, temp_grad_h =  backpropagation(x[indices[i]],y[indices[i]], w1, w2)
        if(i==0):
            grad = temp_grad
            grad_h = temp_grad_h
        else:
            grad = grad + temp_grad
            grad_h = grad_h + temp_grad_h
    grad = grad/5
    grad_h = grad_h/5
    return grad, grad_h


#funcao que calcula o gradiente da funcao de erro quadratico
# utilizando alguns dados antigos
def grad_f_SAGA(x,y,w1,w2,w1_history, w2_history,w_time_hist, n_iter,soma_old, soma_h_old):

    indice = np.random.randint(len(x))
    grad_old, grad_h_old = backpropagation(x[indice],y[indice], w1_history[w_time_hist[indice]], w2_history[w_time_hist[indice]])

    grad_new, grad_h_new = backpropagation(x[indice], y[indice], w1, w2)

    #gradiente = soma_antiga - gradiente_antigo + gradiente_novo
    grad = (soma_old)/5 - grad_old + grad_new
    grad_h = (soma_h_old)/5 - grad_h_old + grad_h_new

    soma_old_return = soma_old - grad_old + grad_new
    soma_h_old_return = soma_h_old - grad_h_old + grad_h_new


    w_time_hist[indice] = n_iter

    return grad, grad_h, soma_old_return, soma_h_old_return

def SAGA(X, y, t):

    # w1 e w2 são matrizes com os pesos da rede neural
    w1 = np.matrix([[1],[2]])
    w2 = np.matrix([3,4])

    n_iter = 0
    erro_quad = 0
    a_history = []
    b_history = []
    soma_old = 0
    soma_h_old = 0
    w1_history = []
    w2_history = []
    w_time_hist = np.zeros(len(X), dtype = int) # guarda a ultima iteração em que cada dado teve seu gradiente calculado

    while True:
        #Na primeira iteração fazemos o método do gradiente
        #nas demais iterações sorteamos um dado e apenas calculamos o gradiente para ele, utilizando
        #o ultimo gradiente calculado para cada amostra
        w1_history.append(w1)
        w2_history.append(w2)


        if n_iter == 0:
            grad, grad_h = grad_f(X, y, w1, w2)
            soma_old = 5*grad
            soma_h_old = 5*grad_h

        else:
            
            grad, grad_h, soma_old, soma_h_old = grad_f_SAGA(X, y, w1, w2, w1_history, w2_history,w_time_hist,n_iter, soma_old, soma_h_old)

        #=========== apenas para plotar o erro==========
        a = n_iter
        a_history.append(a)
        erro_quad = eq(X,y,w1,w2)
        
        b = erro_quad 
        b_history.append(b)
        #===============================================
        
        #anda com passo t na direção oposta à do gradiente
        w1 = w1 - np.multiply(t, grad_h)
        w2 = w2 - np.multiply(t, grad)       

        #norma do gradiente (usado como medida de erros)
        norm_grad = la.norm(np.matrix([[grad_h[0,0]],[grad_h[1,0]],[grad[0,0]],[grad[0,1]]]))
        #condição de parada
        if(norm_grad < 0.0001):
            break

        n_iter = n_iter + 1


#=============== plotando resultados ====================

    # m ideal por quadrados mínimos : 4.09
    m = (w1[0,0]*w2[0,0] + w1[1,0]*w2[0,1])
    print('m: ',  m)
    print('n_iter', n_iter)

    ex = np.linspace(0,6,10)
    ey = m*ex

    plt.axis([0,6,0,30])
    red_dot = plt.scatter(X,[1,4,9,16,25],s = 30, c = "red", label = "dados de treinamento")

    blue_line = plt.plot(ex,ey, label = "y = m*x")

    plt.title("método SAGA\nRegressão de x²: m = {}".format( m))

    plt.legend(loc='upper left')

    plt.show()
    plt.title("Erro quadrático x Tempo")
    plt.xlabel("número de iterações")
    plt.ylabel("erro quadrático")
    plt.scatter(a_history,b_history)
    plt.show()
#========================================================


#aqui temos os dados para treinamento

X = [np.transpose(np.matrix([1.0])),
     np.transpose(np.matrix([2.0])),
     np.transpose(np.matrix([3.0])),
     np.transpose(np.matrix([4.0])),
     np.transpose(np.matrix([5.0]))
     ]

y = [1.0,4.0,9.0,16.0,25.0]

# parâmetros de linha de comando: t 
SAGA(X, y, float(sys.argv[1]) )