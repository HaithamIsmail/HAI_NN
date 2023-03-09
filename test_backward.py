import numpy as np
from HAI_NN import Losses, Activations, Layers
from timeit import timeit
import nnfs

nnfs.init()

softmax_outputs = np.array([[0.7, 0.1, 0.2],
                            [0.1, 0.5, 0.4],
                            [0.02, 0.9, 0.08]])

class_targets = np.array([0, 1, 1])

def f1():
    softmax_loss = Activations.Softmax_With_CategoricalCrossentropy()
    softmax_loss.backward(softmax_outputs, class_targets)
    dvalues1 = softmax_loss.dinputs

def f2():
    acitvation = Activations.Softmax()
    acitvation.output = softmax_outputs
    loss = Losses.CategoricalCrossEntropy()
    loss.backwawrd(softmax_outputs, class_targets)
    acitvation.backward(loss.dinputs)
    dvalues2 = acitvation.dinputs

t1 = timeit(lambda: f1(), number=10000)
t2 = timeit(lambda: f2(), number=10000)

print(t1, t2, t2 / t1)