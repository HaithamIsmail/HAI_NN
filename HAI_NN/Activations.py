import numpy as np
from .Losses import *

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
    
    def backward(self, dvalues):
        
        # create an unintialized array with the same shape as the gradient
        # we are recieving to apply the chain rule
        self.dinputs = np.empty_like(dvalues)
        
        # enumerate outputs and gradients
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            #  flattern output array
            single_output = single_output.reshape(-1, 1)
            
            # calculate jacobian matrix
            # 1. Kronkecker delta equals 1 when the inputs are equal
            # thus a matrix with 1s on the diagonal -> can be done using np.eye
            # softmax_output * np.eye(softmax_output.shape[0])
            # this can be done using a single function np.diagflat
            # 2. The 2nd part of the equation multiplies softmax output
            # by iterating over indices j and k respecitively
            # Since for each sample (j) we'll multiply the values from the softmax
            # function's output (k), we can use dot product by transposing the softmax output
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            
            
            # Calculate sample-wise gradient
            # and add it to the array of sample gradients
            # 1. For a batch, we will have a list of jacobian matrices
            # the derivative of the softmax function with respect to any of its inputs
            # returns a vector of partial derivatives (a row from the jacobian matrix)
            # as the input influences all the outputs (due to normalization process)
            # 2. We need to sum the values of these vectors so that each input for each of
            # the samples will return a single partial derivative value instead. 
            # 3. This operation on each of the jacobian matrices directly
            # applying the chain rule at the same time (applying the gradient from the loss function)
            # using np.dot which will take a row from jacobian matrix and multiply it
            # by a single value from the loss function's gradient, the result is a single value
            # forming a vector of the partial derivatives sample-wise and a 2D array batch-wise.
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)
    
class Softmax_With_CategoricalCrossentropy(activation):
    def __init__(self) -> None:
        self.activation = Softmax()
        self.loss = CategoricalCrossEntropy()
    
    def __call__(self, inputs, y_true):
        return self.forward(inputs, y_true)
    
    def forward (self, inputs, y_true):
        # Output layer's actiation
        self.output = self.activation(inputs)
        
        # Calculate and return loss value
        return self.loss(self.output, y_true)
    
    def backward(self, dvalues: np.ndarray, y_true):
        num_samples = len(dvalues)
        
        # if labels are one hot encoded
        # turn them to discrete
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        
        # copy dvalues to safely modify
        self.dinputs = dvalues.copy()
        # Calculate gradient
        self.dinputs[range(num_samples), y_true] -= 1
        # Normalize gradient
        self.dinputs = self.dinputs / num_samples