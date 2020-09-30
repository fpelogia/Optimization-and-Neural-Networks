import numpy as np
import matplotlib.pyplot as plt
from joelnet.train import train
from joelnet.nn import NeuralNet
from joelnet.layers import Linear, Tanh, Sigmoid, reLu
from joelnet.optim import Optimizer, SGD, Adam
from joelnet.data import BatchIterator
from joelnet.loss import Log_loss

inputs = np.array([
    [0, 0],
    [1, 0],
    [0, 1],
    [1, 1]
])

targets = np.array([
    [0],
    [1],
    [1],
    [0]
])

net = NeuralNet([
    Linear(input_size=2, output_size=4),
    Sigmoid(),
    Linear(input_size=4, output_size=4),
    Sigmoid(),
    Linear(input_size=4, output_size=1),
    Sigmoid()
])

n_epochs = 10000
loss_list = train(net, inputs,targets, loss = Log_loss(),optimizer = SGD(lr = 1e-5),iterator = BatchIterator(4),
 num_epochs = n_epochs)

for x, y in zip(inputs, targets):
    predicted = net.forward(x)
    print(x, predicted, y)

plt.title("Erro quadrático x Tempo")
plt.xlabel("número de iterações")
plt.ylabel("erro quadrático")
plt.scatter(list(range(0, n_epochs)),loss_list)
plt.show()

