from HAI_NN import Layers, Activations, Losses, Optimizers
import nnfs
from nnfs.datasets import spiral_data
import numpy as np

nnfs.init()

X, y = spiral_data(samples=100, classes=2)
print(X.shape)

# reshape because the labels are not (0, 1) and not sparse
y = y.reshape(-1, 1)

dense1 = Layers.Dense(2, 64, weight_regularizer_L2=5e-4, bias_regularizer_L2=5e-4)
activation1 = Activations.ReLU()
# dropout1 =  Layers.Dropout(0.1)
dense2 = Layers.Dense(64, 1)
activation2 = Activations.Sigmoid()
loss_fn = Losses.BinaryCrossentropy()
# optimizer = Optimizers.SGD(decay=1e-3, momentum=0.9)
# optimizer = Optimizers.AdaGrad(decay=1e-4)
# optimizer = Optimizers.RMSprop(learning_rate=0.02,decay=1e-5, rho=0.999)
optimizer = Optimizers.Adam(decay=5e-7)

for epoch in range(10001):
    # Forward prop
    z = dense1(X)
    z = activation1(z)
    z = dense2(z)
    z = activation2(z)
    data_loss = loss_fn(z, y)
    regularization_loss = loss_fn.regularization_loss(dense1) + \
                          loss_fn.regularization_loss(dense2)
    
    loss = data_loss + regularization_loss

    # Calculate the binary mask
    # multiply by 1 to change it to array of 0s and 1s
    prediction = (activation2.output > 0.5) * 1 
    accuracy = np.mean(prediction==y)

    if not epoch % 100:
        print(f'epoch {epoch}, acc: {accuracy: .3f}, loss: {loss: .3f} (data_loss: {data_loss: .3f}, reg_loss: {regularization_loss: .3f}), lr: {optimizer.current_learning_rate}')
        
    # Backward prop
    loss_fn.backward(activation2.output, y)
    activation2.backward(loss_fn.dinputs)
    dense2.backward(activation2.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    optimizer.pre_update_params()
    optimizer(dense1)
    optimizer(dense2)
    optimizer.post_update_params()

# Validate the model
X_test, y_test = spiral_data(samples=100, classes=2)
y_test = y_test.reshape(-1 , 1)
z = dense1(X_test)
z = activation1(z)
z = dense2(z)
z = activation2(z)
loss = loss_fn(z, y_test)

prediction = (activation2.output > 0.5) * 1 
accuracy = np.mean(prediction==y_test)
print(f'validation, acc: {accuracy:.3f}, loss: {loss:.3f}')


