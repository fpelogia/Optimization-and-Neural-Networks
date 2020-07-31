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
import csv


# inputs = np.array([
#     [1],
#     [2],
#     [3],
#     [4],
#     [5]
# ])

#values = np.arange(1,5,0.1)
values = [1.0]
inputs = []
for i in values:
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
    Linear(input_size=2, output_size=1, weights = np.array([[3.0],[4.0]]), biases = np.array([0.0]))
])



n_epochs = 2
eps = 1e-4

#loss_list = train(net, inputs,targets, optimizer = Adam(lr = 1e-2, gamma1 = 0.3, gamma2 = 0.3),iterator = BatchIterator(batch_size = 5), num_epochs = 1000)
loss_list, eval_list = train(net, inputs,targets, loss = MSE() ,optimizer = LM_cond(float(sys.argv[1]),float(sys.argv[2])), iterator = BatchIterator(batch_size =  len(inputs)), num_epochs = n_epochs, eps = eps)





# for x, y in zip(inputs, targets)
#     predicted = net.forward(x)
#     print(x, predicted, y)





ex = np.arange(0,5,0.2)
ey = []
test_loss = []
for val in ex:
    predicted = net.forward([val])
    ey.append(predicted)

#plt.scatter(range(0,len(loss_list)), loss_list)

# plt.title("Regressão de x²")
# plt.axis([0,5,0,25])
# aux = np.arange(6)
# plt.scatter(aux,aux**2,s = 30, c = "red")
# plt.plot(ex,ey)
# #plt.savefig(f'Figuras/Square/EQ.png', format='png')
# plt.show()
plt.title("Regressão de x²")
plt.axis([0,1.1,0,1.1])
aux = np.arange(6)
plt.scatter(aux,aux**2,s = 30, c = "red")
plt.plot(ex,ey)
#plt.savefig(f'Figuras/Square/EQ.png', format='png')
plt.show()

ex = np.linspace(0, len(eval_list), len(eval_list))
print(f'lenex: {len(ex)}\n lenev: {len(eval_list)}')
plt.scatter(ex, loss_list)
plt.axis([0,len(eval_list),0,len(eval_list)])
plt.show()

with open('eval_file.csv', mode='w') as res_file:
	res_writer = csv.writer(res_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
	res_writer.writerow(['Iterações', 'Avaliações', 'Epsilon', '|log(eps)|'])
	for i in range(len(eval_list)):
		res_writer.writerow([i, eval_list[i],loss_list[i], float(abs(np.log(loss_list[i])))])


# print('HAHA1:',eval_list[:10] )
# print('HAHA2:',abs(np.log(loss_list[:10])) )

# print('CORR 0 to 10: ',np.corrcoef(eval_list[:10], abs(np.log(loss_list[:10]))))
# print('CORR 0 to 20: ',np.corrcoef(eval_list[:20], abs(np.log(loss_list[:20]))))
# print('CORR 9 to 30: ',np.corrcoef(eval_list[9:30], abs(np.log(loss_list[9:30]))))
#print('CORR 53 to 69: ',np.corrcoef(eval_list[53:69], abs(np.log(loss_list[53:69]))))
