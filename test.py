from HAI_NN import Layers, Activations, Losses
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

X, y = spiral_data(samples=100, classes=3)
print(X.shape)

dense1 = Layers.Dense(2, 3)
activation1 = Activations.ReLU()
dense2 = Layers.Dense(3, 3)
activation2 = Activations.Softmax()

z = dense1(X)
z = activation1(z)
z = dense2(z)
z = activation2(z)
print(z[:5])

loss = Losses.CategoricalCrossEntropy()

losses = loss(z, y)

print('loss:', losses)
