import numpy as np
from .Activations import *
from .Layers import *

class Loss:
    def __call__(self, output: np.ndarray, y: np.ndarray, *, include_regularization=False):
        return self.calculate(output, y, include_regularization=include_regularization)

    def backward(self, output: np.ndarray, y: np.ndarray):
        pass

    def calculate(self, output: np.ndarray, y: np.ndarray, *, include_regularization=False):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        
        self.accumulated_sum += np.sum(sample_losses)
        self.accumulated_count += len(sample_losses)
        
        if not include_regularization:
            return data_loss
        return data_loss, self.regularization_loss()
    
    def calculate_accumulated(self, *, include_regularization=False):
        data_loss = self.accumulated_sum / self.accumulated_count
        
        if not include_regularization:
            return data_loss
        
        return data_loss, self.regularization_loss()
    
    # Reset variables for accumulated loss
    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0

    def remember_trainable_layers(self, trainable_layers):
        self.trainable_layers: list[Layer] = trainable_layers
    
    def regularization_loss(self):
        regularization_loss = 0
        
        for layer in self.trainable_layers:
            # L1 weights
            if layer.weight_regularizer_L1 > 0:
                regularization_loss += layer.weight_regularizer_L1 * np.sum(np.abs(layer.weights))
            
            # L2 weights
            if layer.weight_regularizer_L2 > 0:
                regularization_loss += layer.weight_regularizer_L2 * np.sum(layer.weights * layer.weights)
            
            # L1 biases
            if layer.bias_regularizer_L1 > 0:
                regularization_loss += layer.bias_regularizer_L1 * np.sum(np.abs(layer.biases))
            
            # L2 biases
            if layer.bias_regularizer_L2 > 0:
                regularization_loss += layer.bias_regularizer_L2 * np.sum(layer.biases * layer.biases)
        
        return regularization_loss
    
class CategoricalCrossEntropy(Loss):
    
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray):        
        # Number of samples in batch
        samples = len(y_pred)
        
        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        
        # Probabilities of target values - 
        # only if categorical labels
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[
                range(samples),
                y_true
            ]
        
        # Mask valeus - only for one-hot encoded labels
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(
                y_pred_clipped * y_true,
                axis=1
            )
        
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    def backwawrd(self, dvalues: np.ndarray, y_true: np.ndarray):
        """
        Backward prop
        params:
            - dvalues: ndarray, predicted values (y-hat)
            - y_true: ndarray, ground-truth vector
        """
        # number of samples
        num_samples = len(dvalues)
        
        # number of labels in every sample
        # We can use the first sample to count them
        num_labels = len(dvalues[0])
        
        # If the labels are sparse, convert them to one-hot vector
        # np.eye returns the identity matrix I with nxn shape and 1s on the diagonal
        if len(y_true.shape) == 1:
            y_true = np.eye(num_labels)[y_true]
            
        # calculate gradient
        self.dinputs = - y_true / dvalues
        
        # normalize gradient
        # optimizers sum all of the gradients related to each weight and bias before multiplying with learning rate
        # so the more number of we have in a dataset, the more gradient sets we'll receive at this step
        # and the larger the sum is. In this case we'll have to adjust the learning rate accoring
        # to the number of samples. Instead, we can solve it by normalizing the gradient and making
        # their sum invariant to the number of samples, as mentioned the optimizer will perform the sum
        self.dinputs = self.dinputs / num_samples

class Softmax_With_CategoricalCrossentropy():
    
    def backward(self, dvalues: np.ndarray, y_true: np.ndarray):
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

class BinaryCrossentropy(Loss):
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray):
        
        # Clip data to prevent division by 0
        # clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 -1e-7)
        
        # Calculate sample-wise loss
        samples_losses = - (y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped)) 
        samples_losses = np.mean(samples_losses, axis=-1)
        
        return samples_losses  
    
    def backward(self, dvalues, y_true):
        # number of samples
        num_samples = len(dvalues)
        
        # number of outputs in each sample
        outputs = len(dvalues[0])
        
        # Clip data to prevent division by 0
        # clip both sides to not drag mean towards any value
        dvalues_clipped = np.clip(dvalues, 1e-7, 1 -1e-7)
        
        # calculate gradient
        # dL/dinputs = 1/J*(y/y_hat)*(1-y / 1-y_hat)
        self.dinputs = - (y_true/dvalues_clipped - (1 - y_true) / (1 - dvalues_clipped)) / outputs
        
        # Normalize gradient
        self.dinputs = self.dinputs / num_samples
    
class MeanSquaredError(Loss):
    
    # forward pass
    def forward(self, y_pred, y_true):
        
        # calculate loss
        # axis -1 to calculate mean across multiple outputs
        sample_losses = np.mean((y_true - y_pred)**2, axis=-1)
        
        # return losses
        return sample_losses
    
    # Backward pass
    def backward(self, dvalues, y_true):
        
        # Number of samples
        num_samples = len(dvalues)
        
        # Number of outputs
        num_outs = len(dvalues[0])
        
        # Calculate gradient w.r.t inputs (i.e. y_hat)
        self.dinputs = -2 * (y_true - dvalues) / num_outs
        
        # normalize to make it invariant to the batch size
        self.dinputs = self.dinputs / num_samples

class MeanAbsoluteError(Loss):
    
    def forward(self, y_pred, y_true):
        
        # calculate loss and mean over all outputs
        samples_losses = np.mean(np.abs(y_true - y_pred), axis=-1)
        
        return samples_losses
    
    def backward(self, dvalues, y_true):
        
        # Number of samples
        num_samples = len(dvalues)
        
        # number of outputs
        num_outs = len(dvalues[0])
        
        # calculate gradients
        self.dinputs = np.sign(y_true - dvalues) / num_outs
        
        # normalize to make it invariant over batch size
        self.dinputs = self.dinputs / num_samples