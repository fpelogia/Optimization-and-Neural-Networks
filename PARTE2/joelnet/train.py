"""
Here's a function that can train a neural net
"""

from joelnet.tensor import Tensor
from joelnet.nn import NeuralNet
from joelnet.loss import Loss, MSE
from joelnet.optim import Optimizer, SGD
from joelnet.data import DataIterator, BatchIterator


def train(net: NeuralNet,
          inputs: Tensor,
          targets: Tensor,
          num_epochs: int = 5000,
          iterator: DataIterator = BatchIterator(),
          loss: Loss = MSE(),
          optimizer: Optimizer = SGD(0.001)) -> [float]:
    
    loss_list = []
    for epoch in range(num_epochs):
        n_iter = 0
        epoch_loss = 0.0
        print(f'================   EPOCH NUMBER {epoch + 1}   ================')
        for batch in iterator(inputs, targets):
            
            #print(f'batch: \n{batch}')
            net.n_iter = n_iter
            net.curr_batch = batch
            net.loss_f = loss
          
            predicted = net.forward(batch.inputs)
            epoch_loss += loss.loss(predicted, batch.targets)
            grad = loss.grad(predicted, batch.targets)
            net.backward(grad)
            optimizer.step(net)
            n_iter = n_iter + 1

        # () / iterator.batch_size
        print(epoch, epoch_loss)
        loss_list.append(epoch_loss)

    return loss_list

