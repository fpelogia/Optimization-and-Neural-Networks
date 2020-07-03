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
n_epochs = 1000
eps = 5
for j in range(5):
	inputs.append([j])
inputs = np.array(inputs)
targets = inputs**2

np.random.seed(2)

net = NeuralNet([
    Linear(input_size=1, output_size=10, weights = np.random.randn(1, 10), biases = np.random.randn(10)),
    reLu(),  
    Linear(input_size=10, output_size=5, weights = np.random.randn(10, 5), biases = np.random.randn(5)),
    reLu(),
    Linear(input_size=5, output_size=1, weights = np.random.randn(5,1), biases = np.random.randn(1))
])

start_time = time.time()
try:
    loss_list, eval_list = train(net, inputs,targets, loss = MSE() ,optimizer = LM_cond(1e6), iterator = BatchIterator(batch_size =  5), num_epochs = n_epochs, eps = eps)
except np.linalg.LinAlgError as err:
    print('Interrompido por matriz singular')
    end_time = time.time()
end_time = time.time()
time_spent = end_time - start_time
print(f'\nt: {time_spent}s')



ex = np.linspace(1,1e3,len(eval_list))
print(f'lenex: {len(ex)}\n lenev: {len(eval_list)}')
plt.scatter(np.log(loss_list), eval_list)
plt.show()


# loglist = []
# for i in range(len(eval_list)):
#     loglist.append(np.log(i))

# print('eval len: ', len(eval_list))
# print('eval l:', eval_list)
# print(np.corrcoef(eval_list, loglist))
