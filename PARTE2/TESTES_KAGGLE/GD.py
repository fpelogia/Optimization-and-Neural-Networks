import numpy as np
from numpy import linalg as la 
import matplotlib.pyplot as plt
import sys
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# data = pd.read_csv("creditcard.csv")

# pd.set_option("display.float", "{:.2f}".format)

# X = data.drop('Class', axis=1)
# y = data.Class

# x_sc = StandardScaler()
# X_std = x_sc.fit_transform(X)

# X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.2, random_state=42)

# X_train = np.array(X_train)
# X_test = np.array(X_test)

# y_train = np.array(y_train)
# y_test = np.array(y_test)





X = [np.matrix([0,0]).T,
	np.matrix([0,1]).T,
	np.matrix([1,0]).T,
	np.matrix([1,1]).T
     ]

X = X*10


w1 = np.matrix([[1 , 2],[3, 4]])
#w1 = np.matrix(np.random.normal(0.0, pow(2, -0.5),(2,1)))
w2 = np.matrix([5,6])
#w2 = np.matrix(np.random.normal(0.0, pow(1, -0.5),(1,2)))

y = [0, 1, 1, 0]
y = y*10


#func = lambda x: max(x,0)
func = lambda x: 1 / (1 + np.exp(-x))
func = np.vectorize(func)

def d_func(x):
	return func(x)*(1 - func(x))
d_func = np.vectorize(d_func)
#     dx = x
#     dx[x<0.0] = 0
#     dx[x>=0.0] = 1
#     return dx



def backpropagation(I, y):

	#print('I:', I)

	H_unac = np.dot(w1,I) # unac indica que é a versao não ativada da camada
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


def feedforward(I):
    H_unac = np.dot(w1,I) # unac indica que é a versao não ativada da camada
    H = func(H_unac)
    O_unac = np.dot(w2,H)
    O = func(O_unac)
    return O





#erro quadrático (funcao a ser minimizada)
def eq(x,y):
	soma = 0
	for i in range(len(x)-1):
		soma = soma + (feedforward(x[i]) - y[i])**2
	return soma/len(x)


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
  grad = grad/len(x)
  grad_h = grad_h/len(x)
  return grad, grad_h



def metodo_gradiente(X, y, t):
    global w1
    global w2
    n_iter = 0
    a_history = []
    b_history = []
    while n_iter < 1e3:
        
        grad, grad_h = grad_f(X, y)
        w1 = w1 - np.multiply(t, grad_h)
        w2 = w2 - np.multiply(t, grad)

        a = n_iter
        a_history.append(a)
        
        
        
        #norm_grad = la.norm(np.matrix([[grad_h[0,0]],[grad_h[1,0]],[grad[0,0]],[grad[0,1]]]))
        b = eq(X,y)
        b_history.append(b)

  
        #if(norm_grad < 0.0001):
        #  break
        n_iter = n_iter + 1

    



    m = (w1[0,0]*w2[0,0] + w1[1,0]*w2[0,1])
    print('m: ',  m)
    print('n_iter', n_iter)


    print('0 0:', feedforward(X[0]))
    print('0 1:', feedforward(X[1]))
    print('1 0:', feedforward(X[2]))
    print('1 1:', feedforward(X[3]))

    # ex = np.linspace(0,6,10)
    # ey = m*ex

    # plt.axis([0,6,0,30])
    # red_dot = plt.scatter(X,[1,4,9,16,25],s = 30, c = "red", label = "dados de treinamento")

    # blue_line = plt.plot(ex,ey, label = "y = m*x")

    # plt.title("Regressão de x²: m = {}".format(m))

    # plt.legend(loc='upper left')

    # plt.show()
    plt.title("Erro quadrático x Tempo")
    plt.xlabel("número de iterações")
    plt.ylabel("erro quadrático")
    plt.scatter(a_history,b_history)
    plt.show()



metodo_gradiente(X, y, float(sys.argv[1]))