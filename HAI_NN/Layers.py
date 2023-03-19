import numpy as np


class Layer:
    def __init__(self, n_inputs, n_neurons,
                 weight_regularizer_L1=0, weight_regularizer_L2=0,
                 bias_regularizer_L1=0, bias_regularizer_L2=0) -> None:
        self.weight_regularizer_L1 = weight_regularizer_L1
        self.weight_regularizer_L2 = weight_regularizer_L2
        self.bias_regularizer_L1 = bias_regularizer_L1
        self.bias_regularizer_L2 = bias_regularizer_L2
        # Initialize weights and biases
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        
        self.prev: Layer = None
        self.next: Layer = None
        self.output: np.ndarray = None
        self.dweights: np.ndarray = None
        self.dbiases: np.ndarray = None
        self.dinputs: np.ndarray = None
                
    def forward(self, inputs, training):
        pass

    def backward(self, dvalues: np.ndarray):
        pass

    def get_parameters(self):
        pass

    def set_parameters(self, weights: np.ndarray, biases: np.ndarray):
        pass
    
    def __call__(self, inputs: np.ndarray, training):
        self.forward(inputs, training)
        return self.output

class Input(Layer):
    def __init__(self):
        pass
    
    def forward(self, inputs: np.ndarray, training):
        self.output = inputs

class Dense(Layer):
    def __init__(self, n_inputs, n_neurons,
                 weight_regularizer_L1=0, weight_regularizer_L2=0,
                 bias_regularizer_L1=0, bias_regularizer_L2=0) -> None:
        Layer.__init__(self, n_inputs, n_neurons,
                       weight_regularizer_L1, weight_regularizer_L2,
                       bias_regularizer_L1, bias_regularizer_L2)
        
    # Retrieve layer parameters
    def get_parameters(self):
        return self.weights, self.biases

    def set_parameters(self, weights: np.ndarray, biases: np.ndarray):
        self.weights = weights
        self.biases = biases
    
    # Forward pass
    def forward(self, inputs: np.ndarray, training):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
    
    # Backward pass
    def backward(self, dvalues: np.ndarray):
        # Gradient for weights and biases (used to update parameters)
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        
        # Gradient on regularization terms
        # L1 weights
        if self.weight_regularizer_L1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.weight_regularizer_L1 * dL1
        # L2 weights
        if self.weight_regularizer_L2 > 0:
            self.dweights += 2 * self.weight_regularizer_L2 * self.weights
        
        # L1 biases
        if self.bias_regularizer_L1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.bias_regularizer_L1 * dL1
        # L2 biases
        if self.bias_regularizer_L2 > 0:
            self.dbiases += 2 * self.bias_regularizer_L2 * self.biases
                    
        # Gradient on inputs (passed to preceeding layer)
        self.dinputs = np.dot(dvalues, self.weights.T)

class Dropout(Layer):
    def __init__(self, rate):
        # Rate argument is the dropout rate
        # we store the inverted rate which is the success rate
        # in the Binomial distribution
        self.rate = 1 - rate
    
    def forward(self, inputs: np.ndarray, training):
        self.inputs = inputs
        
        if not training:
            self.output = inputs.copy()
            return
        
        # Generate scaled mask
        # we add the scaling because we need the inputs
        # and the outputs to have the same scale
        # - Because we are droping some inputs,
        # the sum of the weights * inputs is smaller than it should be
        # by adding the scaling factor we compensate for the dropout
        # otherwise we will have to scale the outputs of the layer down
        self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate
        
        # Apply mask
        self.output = inputs * self.binary_mask
        
    def backward(self, dvalues: np.ndarray):
        self.dinputs = dvalues * self.binary_mask