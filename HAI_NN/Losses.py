import numpy as np

class Loss:
    def __call__(self, output: np.ndarray, y: np.ndarray):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

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
        
        # Mask valeus - only for on-hot encoded labels
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(
                y_pred_clipped * y_true,
                axis=1
            )
        
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    def backwawrd(self, dvalues, y_true):
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