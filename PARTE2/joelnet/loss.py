"""
A loss function measures how good our predictions are,
we can use this to adjust the parameters of our network
"""
import numpy as np

from joelnet.tensor import Tensor

class Loss:
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        raise NotImplementedError

    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        raise NotImplementedError


class MSE(Loss):

    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        #dividir causou instabilidade... rever mais tarde
        #return np.sum((predicted - actual) ** 2)/len(actual)
        return np.sum((predicted - actual) ** 2)

    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        return 2*(predicted - actual)


class Log_loss(Loss):

    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        return -1*np.sum(actual * np.log(predicted) + (np.ones(len(actual)) - actual) * np.log(np.ones(len(actual))-predicted))/len(actual)

    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        if actual == 1.0:
            return -1.0/predicted
        else:
            return 1.0/(1.0 - predicted)

