import numpy as np
from numpy import random
import matplotlib.pyplot as plt


f1 = lambda x: (x[0] - 1)**2 + x[1]**2
f2 = lambda x: x[0]**2 + (x[1] - 1)**2
f3 = lambda x: (x[0] + 1)**2 + x[1]**2
f4 = lambda x: x[0]**2 + (x[1] + 1)**2

f = [f1,f2,f3,f4]

y1 = np.array([1,0])
y2 = np.array([0,1])
y3 = np.array([-1,0])
y4 = np.array([0,-1])

grad1 = lambda x: 2*(x - y1)
grad2 = lambda x: 2*(x - y2)
grad3 = lambda x: 2*(x - y3)
grad4 = lambda x: 2*(x - y4)
grad = [grad1,grad2,grad3,grad4]

#minimizar 0.25*(f1 + f2 + f3 + f4)

#passo
t = 0.1
max_iter = 50

it  = 0
x = np.array([2,3])
historico = []

while it<max_iter:
  historico.append(x)
  
  i = np.random.randint(4) 
  x = x - t*grad[i](x)
  
  
  it = it + 1

ax = [0,0,1,-1]
ay = [1,-1,0,0]

plt.axis([-1.5,4,-2,4])
plt.scatter(ax,ay,s = 30)
bx = []
by = []
for ar in historico:
  bx.append(ar[0])
  by.append(ar[1])
plt.scatter(bx,by, s = 10)
plt.show()
print('solucao encontrada: ',x)