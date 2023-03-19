from . import *
from .Metrics import *
import pickle
from copy import deepcopy
import numpy as np

class ModelNotCompiledException(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)

class Model:
    
    @staticmethod
    def load(path):
        """
        Load pickled model from path
        @params: 
            - path: string, path to pickled model file
        
        @returns: Model object
        """
        with open(path, 'rb') as f:
            model: Model = pickle.load(f)
        return model
    
    def __init__(self, layers: "list[Layers.Layer]" = None) -> None:
        if layers is None:
            self.layers: list[Layers.Layer] = []
        else:
            self.layers: list[Layers.Layer] = layers
        self.softmax_classifier_output = None
        self.compiled = False
        
        self.optimizer: Optimizers.optimizer = None
        self.loss: Losses.Loss = None
        self.accuracy: Accuracies.Accuracy = None
        
        self.loss_history = []
        self.accuracy_history = []
    
    # add layers to the model
    def add(self, Layer: Layers.Layer):
        """
        Add layer to list of layers in the model object
        @params:
            - Layer: Layer object to be added to the list
        
        @returns: None
        """
        self.layers.append(Layer)
        
    def get_parameters(self):
        """
        Get list of parameters (weights and biases) of all layers in the model
        @params: None
        
        @returns: list of tuples of parameters' arrays
        """
        parameters = []
        
        for layer in self.trainable_layers:
            parameters.append(layer.get_parameters)
        
        return parameters
    
    def set_parameters(self, parameters):
        """
        Set the parameters of the layers in the model
        @params: list of parameters
        
        @returns: None
        """
        for parameter_set, layer in zip(parameters, self.trainable_layers):
            layer.set_parameters(*parameter_set)
            
    def save_parameters(self, path):
        """
        Save the list of layers' parameters using pickle package
        @params:
            - path: path where the parameters should be saved
        
        @returns: None
        """
        with open(path, 'wb') as f:
            pickle.dump(self.get_parameters(), f)
            
    def load_parameters(self, path):
        """
        Sets the parameters of layers in the model using file in path
        @params:
            - path: path to the saved parameters file
            
        @returns: None
        """
        with open(path, 'rb') as f:
            self.set_parameters(pickle.load(f))
            
    def save(self, path):
        """
        Saves model object, when loaded the object will retain all the layers and parameters.
        The loss, accuracy, optimizer fields are removed, the model should be compiled again after loading if training or evaluation are to be done
        @params:
            - path: path where the pickled model will be saved
        
        @returns: None
        """
        model = deepcopy(self)
        
        # reset accumulated values
        model.loss.new_pass()
        model.accuracy.new_pass()
        
        # remove data from input layer
        # and gradients from loss object
        model.input_layer.__dict__.pop('output', None)
        
        for layer in model.layers:
            for property in ['inputs', 'output', 'dinputs',
                             'dweights', 'dbiases']:
                layer.__dict__.pop(property, None)
                
        model.loss = None
        model.accuracy = None
        model.optimizer = None
        model.loss_history = []
        model.accuracy_history = []
            
        model.compiled = False
                
        with open(path, 'wb') as f:
            pickle.dump(model, f)
    
    
    # Set loss function and optimizer
    def compile(self, *, loss: Losses.Loss, optimizer: Optimizers.optimizer, accuracy: Accuracies.Accuracy):
        """
        Configure the model for training. Set loss, accuracy, and optimizer then finzalize the model
        @params:
            - loss: loss metric instance
            - optimizer: optimizer instance
            - accuracy: accuracy metric instance, currently supports only one metric
        
        @returns: None
        """
        
        self.loss = loss
        
        self.optimizer = optimizer
        
        self.accuracy = accuracy
    
        self.__finalize()
        
        self.compiled = True
    
    def __finalize(self):
        """
        Connects layers in the layers list, saves trainable layers list and uses optimized layers where available
        """
        # Create and set the input layer
        self.input_layer = Layers.Input()
        
        # number of layers
        layer_count = len(self.layers)
        
        # init a list that will hold the trainable layerss
        self.trainable_layers: list[Layers.Layer] = []
        
        if not isinstance(self.layers[-1], Activations.activation):
            self.layers.append(Activations.Linear())
        
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
                self.output_layer_activation: Activations.activation = self.layers[i]
            
            # If the layer has weights attribute then it is trainable
            # add it to the trainable_layers list
            if hasattr(self.layers[i], 'weights'):
                self.trainable_layers.append(self.layers[i])

        if self.loss is not None:
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

        
    def forward(self, X: np.ndarray, training: bool):
        """
        Perform forward pass on all layers in the model sequentially
        @params:
            - X: samples to perform forward propagation on
            - training: specify if this pass will used for model training
        
        @returns: None
        """
        # Call forward method on input to set its output property
        self.input_layer.forward(X, training)
        
        # Call forward for layer in a chain
        for layer in self.layers:
            layer.forward(layer.prev.output, training=True)
        
        # return last layer output
        return layer.output
    
    def backward(self, output: np.ndarray, y: np.ndarray):
        """
        Perform backward pass and update layers parameters
        @params:
            - output: gradient passed
            - y: ground truth array used to calculate gradient for the loss
        
        @returns: None
        """
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
    
    def train(self, X: np.ndarray, y: np.ndarray, *, epochs=1, batch_size=None, print_every=1, validation_data=None):
        """
        Train model using passed training samples and labels
        @params: 
            - X: training samples
            - y: training samples labels
            - epochs: number of training epochs
            - batch_size: number of samples per gradient update, if None then the whole training set is used
            - print_every: console output update rate, per training step
            - validation_data: tuple containing validation set samples and labels
        
        @returns: None
        """
        if not self.compiled:
            raise ModelNotCompiledException('Model must be compiled before training')
        
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
                    print(f'epoch: {epoch}... {step}/{training_steps}: acc: {accuracy:.3f}, loss: {loss:.3f} (' +
                          f'data loss: {data_loss:.3f}, reg loss: {regularization_loss:.3f}), '+
                          f'lr: {self.optimizer.current_learning_rate}', end='\r')
                
            epoch_data_loss, epoch_regularization_loss = self.loss.calculate_accumulated(include_regularization=True)
            epoch_loss = epoch_data_loss + epoch_regularization_loss
            epoch_accuracy = self.accuracy.calculate_accumulated()
            self.loss_history.append(epoch_loss)
            self.accuracy_history.append(epoch_accuracy)
            
            # Print a summary
            print(f'epoch: {epoch}, acc: {epoch_accuracy:.3f}, ' +
            f'loss: {epoch_loss:.3f} (' +
            f'data_loss: {epoch_data_loss:.3f}, ' +
            f'reg_loss: {epoch_regularization_loss:.3f}), ' +
            f'lr: {self.optimizer.current_learning_rate}')
        
        if validation_data is not None:
            self.evaluate(*validation_data, batch_size=batch_size)            
            

    def evaluate(self, X_val: np.ndarray, y_val: np.ndarray, *, batch_size=None):
        """
        Evaluate model using validation data provided
        @params:
            - X_val: validation data samples
            - y_val: validation data labels
            - batch_size: number of samples per batch of computation
        
        @returns: Return the loss and accuracy values for the model in test mode
        """
        if not self.compiled:
            raise ModelNotCompiledException('Model must be compiled first')
        
        # default number if validation steps
        validation_steps = 1
        # split validation data for better readability
        
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
            self.loss.calculate(output, batch_y)
            
            # get predictions
            predictions = self.output_layer_activation.predictions(output)
            self.accuracy.calculate(predictions, batch_y)
        
        validation_loss = self.loss.calculate_accumulated()
        validation_accuracy = self.accuracy.calculate_accumulated()
        
        # Print a summary
        print(f'validation, ' +
        f'acc: {validation_accuracy:.3f}, ' +
        f'loss: {validation_loss:.3f}')
        
        return validation_loss, validation_accuracy
        
    def predict(self, X: np.ndarray, *, batch_size = None):
        """
        Generate output predictions for input samples
        @params:
            - X: inputs samples
            - batch_size: number of samples per batch of computation
        
        @returns: array of network output
        """
        # default value if batch size is not set
        prediction_steps = 1
        
        if batch_size is not None:
            prediction_steps = len(X) // batch_size
            if prediction_steps * batch_size < len(X):
                prediction_steps += 1

        output = []
        
        for step in range(prediction_steps):
            
            if batch_size is None:
                batch_X = X
            else:
                batch_X = X[prediction_steps*batch_size:(prediction_steps+1)*batch_size]
            
            batch_output = self.forward(batch_X, training=False)
            
            output.append(batch_output)
        
        # stack and return results
        return np.vstack(output)

    def history(self):
        """
        Returns a lists containing the loss and accuracy values over training epochs
        """
        return self.loss_history, self.accuracy_history