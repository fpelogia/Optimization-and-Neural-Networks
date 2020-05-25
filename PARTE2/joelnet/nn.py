"""
A NeuralNet is just a collection of layers.
It behaves a lot like a layer itself, although
we're not going to make it one.
"""
from typing import Sequence, Iterator, Tuple

from joelnet.tensor import Tensor
from joelnet.layers import Layer
import numpy as np
import copy


class NeuralNet:
    def __init__(self, layers: Sequence[Layer]) -> None:
        self.layers = layers
        self.curr_batch = 0
        self.loss_f = 0

    def forward(self, inputs: Tensor) -> Tensor:
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def backward(self, grad: Tensor) -> Tensor:
        for layer in reversed(self.layers):  
            grad = layer.backward(grad)
        return grad

    def params_and_grads(self) -> Iterator[Tuple[Tensor, Tensor]]:

        for layer in self.layers:
            for name, param in layer.params.items():
                grad = layer.grads[name]
                yield param, grad

    def params_and_grads_v2(self) -> Iterator[Tuple[Tensor, Tensor, Tensor]]:

        for layer in self.layers:
            for name, param in layer.params.items():
                grad = layer.grads[name]
                prev = layer.params_prev[name]
                yield param, grad, prev

    def params_and_grads_v3(self) -> Iterator[Tuple[Tensor, Tensor]]:

        for layer in self.layers:
            for name, param in layer.params.items():
                grad = layer.grads[name]
                jac = layer.jacs[name]
                yield param, grad, jac

    def params_and_grads_v4(self) -> Iterator[Tuple[Tensor, Tensor, Tensor]]:

        for layer in self.layers:
            for name, param in layer.params.items():
                grad = layer.grads[name]
                prev = layer.params_prev[name]
                grad_prev = layer.grads_prev[name]
                prev_temp = copy.deepcopy(param)
                yield param, grad, prev, grad_prev
                layer.params_prev[name] = prev_temp