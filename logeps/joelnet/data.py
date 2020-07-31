"""
We'll feed inputs into our network in batches.
So here are some tools for iterating over data in batches.
"""
from typing import Iterator, NamedTuple
from sklearn.utils import shuffle

import numpy as np

from joelnet.tensor import Tensor

Batch = NamedTuple("Batch", [("inputs", Tensor), ("targets", Tensor)])


class DataIterator:
    def __call__(self, inputs: Tensor, targets: Tensor) -> Iterator[Batch]:
        raise NotImplementedError


class BatchIterator(DataIterator):
    def __init__(self, batch_size: int = 32, shuffle: bool = True) -> None:
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __call__(self, inputs: Tensor, targets: Tensor) -> Iterator[Batch]:
        starts = np.arange(0, len(inputs), self.batch_size)

        if self.shuffle:
            inputs, targets = shuffle(inputs, targets, random_state=0)
            #np.random.shuffle(starts)

        #print(f'STARTS: {starts}')

        for start in starts:
            
            end = min(start + self.batch_size, len(targets))
            batch_inputs = inputs[start:end]

            #resolve o problema que estava tendo com o batch_size
            aux = []
            for i in range(start, end):
                try:
                    temp = targets[i][0]
                except IndexError:
                    aux.append([targets[i]])
                else:
                    aux.append([targets[i][0]])

            batch_targets = aux
            #batch_targets = targets[start:end]
            yield Batch(batch_inputs, batch_targets)
