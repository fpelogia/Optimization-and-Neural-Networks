import numpy as np
import matplotlib.pyplot as plt
from joelnet.train import train
from joelnet.nn import NeuralNet
from joelnet.layers import Linear, Tanh, Sigmoid, reLu
from joelnet.data import BatchIterator
from joelnet.optim import SGD, RMSProp, SGD_Nesterov, Adam, Barzilai, LM, GD_cond, LM_cond
from joelnet.loss import MSE, Log_loss
import random
import sys
import time


# inputs = np.array([
#     [1],
#     [2],
#     [3],
#     [4],
#     [5]
# ])


inputs = []
for i in range(20):
	inputs.append([i])
inputs = np.array(inputs)

# targets = np.array([
#     [1],
#     [4],
#     [9],
#     [16],
#     [25]
# ])
targets = inputs**2


net = NeuralNet([
    Linear(input_size=1, output_size=2, weights = np.array([[1.0,2.0]]), biases = np.array([0.0, 0.0])),
    reLu(),
    Linear(input_size=2, output_size=1, weights = np.array([[3.0],[4.0]]), biases = np.array([0.0])),
    reLu()
])



n_epochs = 1000

#loss_list = train(net, inputs,targets, optimizer = Adam(lr = 1e-2, gamma1 = 0.3, gamma2 = 0.3),iterator = BatchIterator(batch_size = 5), num_epochs = 1000)
start_time = time.time()
loss_list = train(net, inputs,targets, loss = MSE() ,optimizer = SGD(1e-5), iterator = BatchIterator(batch_size =  5), num_epochs = n_epochs, eps = 2000)
end_time = time.time()
print(f'Tempo gasto no treinamento: {end_time - start_time}s')




# for x, y in zip(inputs, targets)
#     predicted = net.forward(x)
#     print(x, predicted, y)



#print(f'Levenberg Marquardt com busca linear\nloss = {loss_list[len(loss_list) - 1]:.2f}')

ex = np.linspace(0,20,200)
ey = []
test_loss = []
for val in ex:
    predicted = net.forward([val])
    ey.append(predicted)

plt.title("Erro quadrático x Tempo")
plt.xlabel("número de iterações")
plt.ylabel("erro quadrático")

plt.scatter(list(range(0, n_epochs)),loss_list)
#plt.savefig(f'Figuras/Square/EQ.png', format='png')
plt.show() 

plt.axis([0,20,0,300])
aux = np.arange(21)
plt.scatter(aux,aux**2,s = 30, c = "red")
plt.plot(ex,ey, label = f'Levenberg Marquardt com busca linear\nloss = {loss_list[len(loss_list) - 1]:.02f}')
plt.legend()
#plt.savefig(f'Figuras/Square/REG.png', format='png')
plt.show()


			