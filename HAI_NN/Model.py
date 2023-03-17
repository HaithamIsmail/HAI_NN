from . import *
from .Metrics import *

class Model:
    def __init__(self) -> None:
        self.layers: list[Layers.Layer] = []
        self.softmax_classifier_output = None
    
    # add layers to the model
    def add(self, Layer: Layers.Layer):
        self.layers.append(Layer)
    
    # Set loss function and optimizer
    def compile(self, *, loss: Losses.Loss, optimizer: Optimizers.optimizer, accuracy: Accuracies.Accuracy):
        self.loss = loss
        self.optimizer = optimizer
        self.accuracy = accuracy
    
    def finalize(self):
        
        # Create and set the input layer
        self.input_layer = Layers.Input()
        
        # number of layers
        layer_count = len(self.layers)
        
        # init a list that will hold the trainable layerss
        self.trainable_layers = []
        
        # Iterate over layers
        for i in range(layer_count):
            
            # if first layer, the previous layer is the input layer
            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i+1]
            
            # all other layers excep the last
            elif i < layer_count -1 :
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.layers[i+1]
            
            # last layer is connected to the loss function
            # save a reference to last layer as its output
            # is the model's output
            else:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.loss
                self.output_layer_activation = self.layers[i]
            
            # If the layer has weights attribute then it is trainable
            # add it to the trainable_layers list
            if hasattr(self.layers[i], 'weights'):
                self.trainable_layers.append(self.layers[i])

        # Update loss object with trainable layers
        self.loss.remember_trainable_layers(self.trainable_layers)

        # If output activation is Softmax and
        # loss function is Categorical Cross-Entropy
        # create an object of combined activation
        # and loss function containing
        # faster gradient calculation
        if isinstance(self.layers[-1], Activations.Softmax) and \
        isinstance(self.loss, Losses.CategoricalCrossEntropy):
            # Create an object of combined activation
            # and loss functions
            self.softmax_classifier_output = Losses.Softmax_With_CategoricalCrossentropy()

        
    def forward(self, X, training):
        
        # Call forward method on input to set its output property
        self.input_layer.forward(X, training)
        
        # Call forward for layer in a chain
        for layer in self.layers:
            layer.forward(layer.prev.output, training=True)
        
        # return last layer output
        return layer.output
    
    def backward(self, output, y):
        
        # if softmax classifier
        if self.softmax_classifier_output is not None:
            # call backward method
            # on the optimized acitvation/loss
            # this will set the dinputs property
            self.softmax_classifier_output.backward(output, y)
            
            # we already have dinputs of last layer
            self.layers[-1].dinputs = self.softmax_classifier_output.dinputs
            
            # call backward for all layers except for the last
            for layer in reversed(self.layers[:-1]):
                layer.backward(layer.next.dinputs)
            return
        
        # Calculate backward pass for loss
        # to set dinputs property that the last layer will access
        self.loss.backward(output, y)

        # Call the backward method for all layers
        # in reverse order passing dinputs parameter
        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)
    
    def train(self, X, y, *, epochs=1, batch_size=None, print_every=1, validation_data=None):
        self.accuracy.init(y)
        
        # Default number of steps if batch size is not set
        training_steps = 1
        
        # calculate number of steps if batch size is not none
        if batch_size is not None:
            training_steps = len(X) // batch_size
            
            # add one more step to include extra sample not included in batches
            if training_steps * batch_size < len(X):
                training_steps += 1
                
        # Main training loop
        for epoch in range(1, epochs+1):
            
            print(f'epoch: {epoch}')
            
            # reset accumulated values for loss and accuracy
            self.loss.new_pass()
            self.accuracy.new_pass()
            
            # iterate over training steps
            for step in range(training_steps):
                
                # If batch size is not set -
                # training on the full training set
                if batch_size is None:
                    batch_X = X
                    batch_y = y
                
                # otherwise slice into batches
                else:
                    batch_X = X[step*batch_size:(step+1)*batch_size]
                    batch_y = y[step*batch_size:(step+1)*batch_size]
                
                # perform forward pass for all layers
                output = self.forward(batch_X, training=True)
                
                data_loss, regularization_loss = self.loss(output, batch_y, include_regularization=True)
                loss = data_loss + regularization_loss
                
                predictions = self.output_layer_activation.predictions(output)
                
                accuracy = self.accuracy.calculate(predictions, batch_y)
                
                # perform backward pass for all layers
                self.backward(output, batch_y)
                
                # update learning rate
                self.optimizer.pre_update_params()
                # updata parameters for each layer
                for layer in self.trainable_layers:
                    self.optimizer.update_params(layer)
                # increase iteration counter
                self.optimizer.post_update_params()
                
                # print summary for step
                if not step % print_every or step == training_steps - 1:
                    print(f'step: {step}, acc: {accuracy:.3f}, loss: {loss:.3f} (' +
                          f'data loss: {data_loss:.3f}, reg loss: {regularization_loss:.3f}), '+
                          f'lr: {self.optimizer.current_learning_rate}')
                
                epoch_data_loss, epoch_regularization_loss = self.loss.calculate_accumulated(include_regularization=True)
                epoch_loss = epoch_data_loss + epoch_regularization_loss
                epoch_accuracy = self.accuracy.calculate_accumulated()
                
                # Print a summary
                print(f'acc: {epoch_accuracy:.3f}, ' +
                f'loss: {epoch_loss:.3f} (' +
                f'data_loss: {epoch_data_loss:.3f}, ' +
                f'reg_loss: {epoch_regularization_loss:.3f}), ' +
                f'lr: {self.optimizer.current_learning_rate}')
        
        if validation_data is not None:
            
            # default number if validation steps
            validation_steps = 1
            # split validation data for better readability
            X_val, y_val = validation_data
            
            if batch_size is not None:
                validation_steps = len(X_val) // batch_size
                if validation_steps * batch_size < len(X_val):
                    validation_steps += 1
            
            # reset accumulated values in loss and accuracy
            
            for step in range(validation_steps):
                
                # if batch size is not set -
                # use the full validation set
                if batch_size is None:
                    batch_X = X_val
                    batch_y = y_val
                # Otherwise slice
                else:
                    batch_X = X_val[step*batch_size:(step+1)*batch_size]
                    batch_y = y_val[step*batch_size:(step+1)*batch_size]
                
                # forward pass
                output = self.forward(batch_X, training=False)
                
                # calculate loss
                loss = self.loss.calculate(output, batch_y)
                
                # get predictions
                predictions = self.output_layer_activation.predictions(output)
                self.accuracy.calculate(predictions, batch_y)
            
            validation_loss = self.loss.calculate_accumulated()
            validation_accuracy = self.accuracy.calculate_accumulated()
            
            # Print a summary
            print(f'validation, ' +
            f'acc: {validation_accuracy:.3f}, ' +
            f'loss: {validation_loss:.3f}')
