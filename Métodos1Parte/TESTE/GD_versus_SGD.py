import numpy as np
from numpy import linalg as la 
import matplotlib.pyplot as plt
import sys
import random
import time
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





# #erro quadrático (funcao a ser minimizada)
# def eq(x,y,w1,w2):
#     parte_1 = (feedforward(x[0],w1,w2) - y[0])**2
#     parte_2 = (feedforward(x[1],w1,w2) - y[1])**2
#     parte_3 = (feedforward(x[2],w1,w2) - y[2])**2
#     parte_4 = (feedforward(x[3],w1,w2) - y[3])**2
#     parte_5 = (feedforward(x[4],w1,w2) - y[4])**2

#     return (parte_1 + parte_2 + parte_3 + parte_4 + parte_5)/5


#funcao que calcula o gradiente da funcao de erro quadratico
def grad_f(x,y, w1, w2):
  temp_grad = 0
  temp_grad_h = 0
  for i in range(len(x)):
    temp_grad, temp_grad_h =  backpropagation(x[i],y[i],w1,w2)
    if(i==0):
      grad = temp_grad
      grad_h = temp_grad_h
    else:
      grad = grad + temp_grad
      grad_h = grad_h + temp_grad_h
  grad = grad/len(x)
  grad_h = grad_h/len(x)
  return grad, grad_h

  #funcao que calcula o gradiente estocástico
def grad_f_SGD(x,y,w1,w2,batch_size):
  temp_grad = 0
  temp_grad_h = 0
  indices = random.sample(range(0, len(x)), batch_size)
  for i in range(batch_size):
    temp_grad, temp_grad_h =  backpropagation(x[indices[i]],y[indices[i]], w1, w2)
    if(i==0):
      grad = temp_grad
      grad_h = temp_grad_h
    else:
      grad = grad + temp_grad
      grad_h = grad_h + temp_grad_h
  grad = grad/batch_size
  grad_h = grad_h/batch_size
  return grad, grad_h

def MetodoGradiente(X, y, t):

    # w1 e w2 são matrizes com os pesos da rede neural
    w1 = np.matrix([[1],[2]])
    w2 = np.matrix([3,4])

    n_iter = 0
    erro_quad = 0
    # a_history = []
    # b_history = []

    while n_iter < 10000:
        
        #descobre o gradiente
        grad, grad_h = grad_f(X, y, w1, w2)


        #=========== apenas para plotar o erro==========
        # a = n_iter
        # a_history.append(a)
        # erro_quad = eq(X,y,w1,w2)
        # b = erro_quad 
        # b_history.append(b)
        #===============================================
        

        #anda com passo t na direção oposta à do gradiente
        w1 = w1 - np.multiply(t, grad_h)
        w2 = w2 - np.multiply(t, grad)        
        
        #norma do gradiente (usado como medida de erros)
        norm_grad = la.norm(np.matrix([[grad_h[0,0]],[grad_h[1,0]],[grad[0,0]],[grad[0,1]]]))

        m = (w1[0,0]*w2[0,0] + w1[1,0]*w2[0,1])
        if m < 4.15 and m > 4.03:
          break
        
        
        # #condição de parada
        # if(norm_grad < 0.0001):
        #   break

        n_iter = n_iter + 1


#=============== resultados ====================

    # m ideal por quadrados mínimos : 4.09
    m = (w1[0,0]*w2[0,0] + w1[1,0]*w2[0,1])
    print('m: ',  m)
    print('n_iter', n_iter)



# MÉTODO DO GRADIENTE ESTOCÁSTICO
#========================================================
def SGD(X, y, t, batch_size = 5):

    # w1 e w2 são matrizes com os pesos da rede neural
    w1 = np.matrix([[1],[2]])
    w2 = np.matrix([3,4])

    n_iter = 0
    erro_quad = 0
    # a_history = []
    # b_history = []

    while n_iter < 10000:
        
        #descobre o gradiente
        grad, grad_h = grad_f_SGD(X, y, w1, w2, batch_size)


        #=========== apenas para plotar o erro==========
        # a = n_iter
        # a_history.append(a)
        # erro_quad = eq(X,y,w1,w2)
        # b = erro_quad 
        # b_history.append(b)
        #===============================================
        

        #anda com passo t na direção oposta à do gradiente
        w1 = w1 - np.multiply(t, grad_h)
        w2 = w2 - np.multiply(t, grad)        
        
        #norma do gradiente (usado como medida de errosi)
        norm_grad = la.norm(np.matrix([[grad_h[0,0]],[grad_h[1,0]],[grad[0,0]],[grad[0,1]]]))

        m = (w1[0,0]*w2[0,0] + w1[1,0]*w2[0,1])
        if m < 4.15 and m > 4.03:
          break
        
        #condição de parada
        # if(norm_grad < 0.0001):
        #   break

        n_iter = n_iter + 1

#=============== resultados ====================

    # m ideal por quadrados mínimos : 4.09
    m = (w1[0,0]*w2[0,0] + w1[1,0]*w2[0,1])
    print('m: ',  m)
    print('n_iter', n_iter)


#========================================================



#aqui temos os dados para treinamento

X = []
y = []
aux = [1,2,3,4,5]
aux = 2000*aux
random.shuffle(aux)

for i in range(len(aux)):
  val = aux[i]
  X.append(np.transpose(np.matrix([val])))
  y.append(val**2)
print("\n Treinando com 10000 dados")
# sns.distplot(np.sqrt(y))
# plt.show()
print("\n===== Método do gradiente =====")
start_time = time.time()
MetodoGradiente(X, y, float(sys.argv[1]))
end_time = time.time()
print(f'Tempo de execução: {end_time - start_time}s')
print("====================================")
print(f"\n\n===== SGD com batch_size {int(sys.argv[2])} =====")
start_time = time.time()
SGD(X, y, float(sys.argv[1]), int(sys.argv[2]))
end_time = time.time()
print(f'Tempo de execução: {end_time - start_time}s')
print("====================================\n\n")
