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

inputs = []
n_epochs = 10000
eps = 5
for j in range(int(5)):
	inputs.append([j])
inputs = np.array(inputs)
targets = inputs**2

np.random.seed(20)

net = NeuralNet([
    Linear(input_size=1, output_size=2, weights = np.random.randn(1, 2), biases = np.random.randn(2)),
    reLu(),  
    Linear(input_size=2, output_size=2, weights = np.random.randn(2, 2), biases = np.random.randn(2)),
    reLu(),
    Linear(input_size=2, output_size=1, weights = np.random.randn(2,1), biases = np.random.randn(1))
])

start_time = time.time()
try:
    loss_list, eval_list = train(net, inputs,targets, loss = MSE() ,optimizer = LM_cond(1e15), iterator = BatchIterator(batch_size =  5), num_epochs = n_epochs, eps = eps)
except np.linalg.LinAlgError as err:
    print('Interrompido por matriz singular')
    end_time = time.time()
end_time = time.time()
time_spent = end_time - start_time
print(f'\nt: {time_spent}s')



ex = np.linspace(0, n_epochs,n_epochs)
print(f'lenex: {len(ex)}\n lenev: {len(eval_list)}')
plt.scatter(ex, abs(np.log(loss_list)))
plt.axis([0,n_epochs,0,n_epochs])
plt.show()

print(np.corrcoef(eval_list, abs(np.log(loss_list))))
