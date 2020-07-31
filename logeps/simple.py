import numpy as np
import matplotlib.pyplot as plt
import sys

w = np.matrix([1,2,3,4]).T

def forward (w,x):
  return (w[0][0,0]*w[2][0,0] + w[1][0,0]*w[3][0,0])*x

def F(w,x,y):
  return forward(w,x) - y

def f(w,x,y):
  return 0.5*(F(w,x,y))**2

def JF(w):
  return np.matrix([w[2][0,0],w[3][0,0],w[0][0,0],w[1][0,0]])

def gf(w,x,y):
  return F(w,x,y)*JF(w).T

def LMBL (w,x,y,alpha):
  loss = []
  neval = []
  eps = 1e-3
  for i in range(10000):
    lamb = np.linalg.norm(gf(w,x,y))
    d = np.linalg.solve(JF(w).T @ JF(w) + lamb*np.eye(len(w)), -1*gf(w,x,y) )
    count = 0
    while not f(w+d,x,y) < f(w,x,y) - alpha*np.linalg.norm(d)**2:
      lamb = 2*lamb
      d = np.linalg.solve(JF(w).T @ JF(w) + lamb*np.eye(len(w)), -1*gf(w,x,y) )
      count = count + 1
    w = w + d
    fval = f(w,x,y)
    loss.append(f(w,x,y))
    if(len(neval)== 0):
      neval.append(1 + count)
    else:
      neval.append(neval[len(neval)-1] + 1 + count)

    if fval < eps:
      print(f'precisão atingida após {i} iterações')
      break
  return loss, neval
    
  

 
loss, neval = LMBL(w,1,1,1e2)
#plt.plot(loss)

print('CORRELAÇÃO ENTRE NÚMERO AVALIAÇÕES DE FUNÇÃO E |LOG eps| \nA partir da iteração 11:\n',np.corrcoef(neval[11:], abs(np.log(loss[11:]))))

    
  

  



















