import numpy as np

class activation:
    def __call__(self, inputs):
        self.forward(inputs)
        return self.output

class ReLU(activation):
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
    
    def backward(self, dvalues: np.ndarray):
        # Make a copy to make sure the original variable is not modified
        self.dinputs = dvalues.copy()
        
        # Zero gradient where input values are negative
        self.dinputs[self.inputs <= 0] = 0
        
    
class Softmax(activation):
    def forward(self, inputs):
        
        # Get unnormalized probabilites
        # Subtract by the max to avoid exploding values
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        
        # Normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        
        self.output = probabilities