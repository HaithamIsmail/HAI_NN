import numpy as np

class activation:
    def __call__(self, inputs: np.ndarray, training=False):
        self.forward(inputs, training)
        return self.output

class Linear(activation):
    def forward(self, inputs: np.ndarray, training):
        self.inputs = inputs
        self.output = inputs
    
    def backward(self, dvalues: np.ndarray):
        self.dinputs = dvalues.copy()

    # Calculate prediction for outputs
    def predictions(self, outputs: np.ndarray):
        return outputs
        
class ReLU(activation):
    def forward(self, inputs: np.ndarray, training):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
    
    def backward(self, dvalues: np.ndarray):
        # Make a copy to make sure the original variable is not modified
        self.dinputs = dvalues.copy()
        
        # Zero gradient where input values are negative
        self.dinputs[self.inputs <= 0] = 0
    
    # Calculate prediction for outputs
    def predictions(self, outputs: np.ndarray):
        return outputs

class Sigmoid(activation):
    def forward(self, inputs: np.ndarray, training):
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))
    
    def backward(self, dvalues: np.ndarray):
        # Derivative w.r.t to inputs is sigma * (1-sigma)
        # dJ/dinputs = dJ/dz * dz/dinputs 
        self.dinputs = dvalues * (1 - self.output) * self.output
    
    # Calculate predictions
    # multiply by 1 to tranform values from (True/False) to (1/0)
    def predictions(self, outputs: np.ndarray):
        return (outputs > 0.5) * 1
    
class Softmax(activation):
    def forward(self, inputs: np.ndarray, training):
        
        self.inputs = inputs
        
        # Get unnormalized probabilites
        # Subtract by the max to avoid exploding values
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        
        # Normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        
        self.output = probabilities
    
    def backward(self, dvalues: np.ndarray):
        
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
    
    def predictions(self, outputs: np.ndarray):
        return np.argmax(outputs, axis=1)