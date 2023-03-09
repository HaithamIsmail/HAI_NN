from HAI_NN import Layers, Activations, Losses, Optimizers
import nnfs
from nnfs.datasets import spiral_data
import numpy as np

nnfs.init()

X, y = spiral_data(samples=100, classes=3)
print(X.shape)

dense1 = Layers.Dense(2, 64)
activation1 = Activations.ReLU()
dense2 = Layers.Dense(64, 3)
activation_loss = Activations.Softmax_With_CategoricalCrossentropy()
# optimizer = Optimizers.SGD(decay=1e-3, momentum=0.9)
# optimizer = Optimizers.AdaGrad(decay=1e-4)
# optimizer = Optimizers.RMSprop(learning_rate=0.02,decay=1e-5, rho=0.999)
optimizer = Optimizers.Adam(learning_rate=0.05, decay=5e-7)

for epoch in range(10001):
    # Forward prop
    z = dense1(X)
    z = activation1(z)
    z = dense2(z)
    loss = activation_loss(z, y)

    prediction = np.argmax(activation_loss.output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    accuracy = np.mean(prediction==y)

    if not epoch % 100:
        print(f'epoch {epoch}, acc: {accuracy: .3f}, loss: {loss: .3f}, lr: {optimizer.current_learning_rate}')
    # Backward prop
    activation_loss.backward(activation_loss.output, y)
    dense2.backward(activation_loss.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    optimizer.pre_update_params()
    optimizer(dense1)
    optimizer(dense2)
    optimizer.post_update_params()


