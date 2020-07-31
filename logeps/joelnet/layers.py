from typing import Dict, Callable

import numpy as np

from joelnet.tensor import Tensor
from joelnet.jacobian import jac_builder


class Layer:
    def __init__(self) -> None:
        self.params: Dict[str, Tensor] = {}
        self.params_prev: Dict[str, Tensor] = {}
        self.grads: Dict[str, Tensor] = {}
        self.grads_prev: Dict[str, Tensor] = {}
        self.jacs: Dict[str, Tensor] = {}

    def forward(self, inputs: Tensor) -> Tensor:
        """
        Produce the outputs corresponding to these inputs
        """
        raise NotImplementedError

    def backward(self, grad: Tensor) -> Tensor:
        """
        Backpropagate this gradient through the layer
        """
        raise NotImplementedError


class Linear(Layer):
    """
    computes output = inputs @ w + b
    """
    def __init__(self,input_size: int, output_size: int, weights: Tensor, biases: Tensor) -> None:
        # inputs will be (batch_size, input_size)
        # outputs will be (batch_size, output_size)
        super().__init__()
        # self.params["w"] = np.random.randn(input_size, output_size)
        # self.params["b"] = np.random.randn(output_size)
        self.params["w"] = weights
        self.params["b"] = biases
        self.params_prev["w"] = np.zeros((input_size, output_size))
        self.params_prev["b"] = np.zeros(output_size)
        self.grads_prev["w"] = np.zeros((input_size, output_size))
        self.grads_prev["b"] = np.zeros(output_size)
        self.grads["w"] = np.zeros((input_size, output_size))
        self.grads["b"] = np.zeros(output_size)
        


    def forward(self, inputs: Tensor) -> Tensor:
        """
        outputs = inputs @ w + b
        """
        self.inputs = inputs
        return inputs @ self.params["w"] + self.params["b"]

    def backward(self, grad: Tensor) -> Tensor:
        """
        if y = f(x) and x = a * b + c
        then dy/da = f'(x) * b
        and dy/db = f'(x) * a
        and dy/dc = f'(x)

        if y = f(x) and x = a @ b + c
        then dy/da = f'(x) @ b.T
        and dy/db = a.T @ f'(x)
        and dy/dc = f'(x)
        """

        self.grads_prev["b"] = self.grads["b"]
        self.grads["b"] = np.sum(grad, axis = 0)
        self.jacs["b"] = grad
        #print(f'INPUTS: {self.inputs.shape} \n GRAD: {grad.shape}')
        #print(f'JAC: {jac_builder(self.inputs, grad)}')
        #print(f'It@G: {self.inputs.T@grad}')
        #print(f'INPSHP: {self.inputs.shape} GDSHP: {grad.shape}')
        self.grads_prev["w"] = self.grads["w"]
        print('GRADD: ', grad)
        print('SGW: ', self.inputs.T @ grad)
        self.grads["w"] = self.inputs.T @ grad
        #print( f'JBIN: {self.inputs}\nJBGD: {grad}')
        self.jacs["w"] = jac_builder(self.inputs, grad)

        #print(f'gradw.shape: {self.grads["w"].shape}\n jtjw.shape: {(self.jacs["w"].T@self.jacs["w"]).shape}')
        #print(f'gradw.shape: {self.grads["b"].shape}\n jtjw.shape: {(self.jacs["b"].T@self.jacs["b"]).shape}')
        #gdwb = np.column_stack((self.jacs["w"], self.jacs["b"] )) 
        return grad @ self.params["w"].T


F = Callable[[Tensor], Tensor]

class Activation(Layer):
    """
    An activation layer just applies a function
    elementwise to its inputs
    """
    def __init__(self, f: F, f_prime: F) -> None:
        super().__init__()
        self.f = f
        self.f_prime = f_prime

    def forward(self, inputs: Tensor) -> Tensor:
        self.inputs = inputs
        return self.f(inputs)

    def backward(self, grad: Tensor) -> Tensor:
        """
        if y = f(x) and x = g(z)
        then dy/dz = f'(x) * g'(z)
        """
        return self.f_prime(self.inputs) * grad


def tanh(x: Tensor) -> Tensor:
    return np.tanh(x)

def tanh_prime(x: Tensor) -> Tensor:
    y = tanh(x)
    return 1 - y ** 2

def sigmoid(x: Tensor) -> Tensor:
    return 1/(1+np.exp(-1*x)) 

def sigmoid_prime(x: Tensor) -> Tensor:
    return sigmoid(x)*(1 - sigmoid(x))

epsilon = 0.0001

def relu(x: Tensor) -> Tensor:
    return 0.5*(x + np.sqrt(x**2 + epsilon))
    
relu = np.vectorize(relu)

def relu_prime(x: Tensor) -> Tensor:
    return 0.5*(x/np.sqrt(x**2 + epsilon) + 1.0)


class Tanh(Activation):
    def __init__(self):
        super().__init__(tanh, tanh_prime)


class Sigmoid(Activation):
    def __init__(self):
        super().__init__(sigmoid, sigmoid_prime)


class reLu(Activation):
    def __init__(self):
        super().__init__(relu, relu_prime)
