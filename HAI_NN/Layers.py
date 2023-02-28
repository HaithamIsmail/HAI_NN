import numpy as np


class Layer:
    def __call__(self, inputs):
        self.forward(inputs)
        return self.output
    
class Dense(Layer):
    def __init__(self, n_inputs, n_neurons) -> None:
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
    
    def backward(self, dvalues: np.ndarray):
        # Gradient for weights and biases (used to update parameters)
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        
        # Gradient on inputs (passed to preceeding layer)
        self.dinputs = np.dot(dvalues, self.weights.T)