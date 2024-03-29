# HAI_NN

**A neural network framework using Numpy package**

This is my attempt to get a deeper understanding of neural networks down to its lowest levels. The code was written with the help of [Neural Networks from Scratch](https://nnfs.io/) book (definitely recommended). 

I am trying to add my own touches by building it into a package like structure, and introducing some functionalities not available in the book. Hopefully, I will write the whole package in my own code and publish it someday.

# Install

Clone reqpository or download it as a zip then unpack

```
git clone https://github.com/HaithamIsmail/HAI_NN.git
```

To install, open the module folder in terminal and use:

```
python setup.py install
```


# Testing

A jupyter notebook "[Showcase.ipynb](Showcase.ipynb)" is available to showcase the module.

A model is built and trained on Fashion mnist dataset reaching 93.1% training accuracy and 92.2% validation accuracy. At the end, out-of-sample images downloaded from the internet are used as testing samples where it correctly predict 8/9 labels.

# Documentation

## Layers

The framework includes the following types of layers:

### Dense layer: 

This layer represents a fully connected layer in a neural network, where each neuron in the current layer is connected to all neurons in the previous layer. The output of each neuron is calculated as a weighted sum of the inputs followed by a bias term.

```py
HAI_NN.Layers.Dense(
    n_inputs,
    n_neurons,
    weight_regularizer_L1=0,
    weight_regularizer_L2=0,
    bias_regularizer_L1=0,
    bias_regularizer_L2=0
)
```

Args:

1. n_inputs: number of input featres
2. n_neurons: number of neurons in the layer
3. weight_regularizer_L1: weights L1 regularization scaling factor
4. weight_regularizer_L2: weights L2 regularization scaling factor
5. bias_regularizer_L1: biases L1 regularization scaling factor
6. bias_regularizer_L2: biases L2 regularization scaling factor

### Dropout:

This layer is used to prevent overfitting in deep learning models. It randomly drops out a fraction of the neurons in the previous layer during training, forcing the remaining neurons to learn more robust and generalizable features. Inputs not set to 0 are scaled up by 1/(1 - rate) such that the sum over all inputs is unchanged

```py
HAI_NN.Layers.Dropout(
    rate
)
```

Args:

1. rate: fraction between 0 and 1. Fraction of the input units to drop

## Activations

The framework includes four types of activation functions, activations are a subclass of the Layer class:

### Linear:

The output is simply the input, with no nonlinearity applied. 

```py
HAI_NN.Activations.Linear()
```

Methods

**predictions**

Returns the output argument

```py
predictions(
    outputs
)
```

#

### ReLU:

The output is the maximum of the input and zero , providing a simple and effective way 
to introduce nonlinearity into the model.

$$ ReLU(x) = max(x, 0) $$

```py
HAI_NN.Activations.ReLU()
```

Methods

**predictions**

Returns the output argument

```py
predictions(
    outputs
)
```

#

#### Softmax:

This activation is typically used for multi-class classification problems, where the output of each neuron represents the probability of belonging to a particular class. The softmax function ensures that the output probabilities sum to one. The softmax of a vector **x** calculated using:

$$ Softmax(x) = {e^x \over \sum e^x} $$

```py
HAI_NN.Activations.Softmax()
```

Methods

**predictions**

Returns index of class with highest confidence.

(outputs > 0.5) * 1

```py
predictions(
    outputs
)
```

#

### Sigmoid:

This activation is typically used for binary classification problems, where the output of each neuron represents the probability of belonging to the positive class. The sigmoid function ensures that the output is between zero and one. 

$$ sigmoid(x) = 1 / (1 + e^{-x}) $$

```py
HAI_NN.Activations.Sigmoid()
```
Methods

**predictions**

Get class prediction for outputs. If output is larger than 0.5 then it is of label "1".

(outputs > 0.5) * 1

```py
predictions(
    outputs
)
```


#

## Losses

The framework includes four types of loss functions:

### Binary Crossentropy:

This loss is typically used for binary classification problems, where the model outputs a probability of belonging to the positive class. The binary crossentropy loss measures the difference between the predicted probability and the true label.

The loss function requires the following inputs:
1. y_pred: predicted labels vector
2. y_true: ground truth (true labels) vector

```py
HAI_NN.Losses.BinaryCrossentropy()
```

**Recommended Usage**

```py
model.compile(
    loss=HAI_NN.Losses.BinaryCrossentropy(),
    ...
)
```

### Categorical Crossentropy: 

This loss is typically used for multi-class classification problems, where the model outputs a probability distribution over multiple classes. The categorical crossentropy loss measures the difference between the predicted distribution and the true distribution of class labels. Sparse and one-hot encoded inputs are both accepted.

The loss function requires the following inputs:
1. y_pred: predicted labels vector
2. y_true: ground truth (true labels) vector

```py
HAI_NN.Losses.CategoricalCrossEntropy()
```

**Recommended Usage**

```py
model.compile(
    loss=HAI_NN.Losses.CategoricalCrossEntropy(),
    ...
)
```

### Mean Squared Error:

This loss is typically used for regression problems, where the model outputs a continuous value. The mean squared error loss measures the difference between the predicted value and the true value.

$$ cost = (y_{true} - y_{pred})^2 $$

```py
HAI_NN.Losses.MeanSquaredError()
```

**Recommended Usage**

```py
model.compile(
    loss=HAI_NN.Losses.MeanSquaredError(),
    ...
)
```

### Mean Absolute Error:

This loss is similar to the mean squared error loss, but measures the absolute difference between the predicted value and the true value.

$$ cost = |y_{true} - y_{pred}| $$

```py
HAI_NN.Losses.MeanAbsoluteError()
```

**Recommended Usage**

```py
model.compile(
    loss=HAI_NN.Losses.MeanAbsoluteError(),
    ...
)
```

## Optimizers

### Stochastic Gradient Descent (with momentum):

This optimizer updates the parameters of the model in the direction of the negative gradient of the loss function.

```py
HAI_NN.Optimizers.SGD(
    learning_rate=1.,
    decay=0.,
    momentum=0.
)
```

Args:

1. learning_rate: initial learning rate
2. decay: decay rate
3. momentum: hyperparameter that accelerate gradient descent in the relevant direction and dampens oscillations

if decay is specified, learning rate is updated using the following rule:

$$ lr_{current} = lr \times { 1 \over (1 + decay \times iterations)} $$

Update rule where **paramter** can be weights or biases:

1. When momentum is 0:
   
$$ parameter = parameter - lr_{current} \times gradient $$

1. when momentum is larger than zero:
   
$$ momentums_{parameter} = momentum \times momentums_{parameter} - lr_{current} \times gradient $$

$$ parameter = parameter + momentums_{parameter} $$

#

### AdaGrad:
This optimizer adapts the learning rate of each parameter based on the historical gradient information, in order to converge faster on flat directions of the loss surface.

```py
HAI_NN.Optimizers.AdaGrad(
    learning_rate=1.,
    decay=0.,
    epsilon=1e-7
)
```

Args:

1. learning_rate: initial learning rate
2. decay: learning rate decay rate
3. epsilon: small floating point used to maintain numerical stability

Update rule where **parameter** is either weights or biases:

$$ parameter = parameter - {lr_{current} \times gradient \over (\sqrt{\sum{gradient^2}} + epsilon)} $$

#

### RMSprop:

This optimizer also adapts the learning rate of each parameter based on the historical gradient information, but uses a moving average of the squared gradients to stabilize the learning rate. RMSprop works by keeping an expoenentially weighted average of the square of the gradients, and then dividing square root of this average.

```py
HAI_NN.Optimizers.RMSprop(
    learning_rate=0.001,
    decay=0.,
    epsilon=1e-7,
    rho=0.9
)
```

Args:

1. learning_rate: initial learning rate
2. decay: learning rate decay rate
3. epsilon: small floating point used to maintain numerical stability
4. rho: Discounting factor for the history/coming gradient

Update rule:

$$ cache_{parameter} = rho \times cache_{parameter} + (1 - rho) \times gradient^2 $$

$$ parameter = parameter - lr_{current} \times { gradient \over (\sqrt{cache_{parameter}} + epsilon)} $$

#

### Adam:
This optimizer combines the ideas of momentum and adaptive learning rates, by maintaining a moving average of the gradient and the squared gradient, and using them to update the parameters with a scaled learning rate. (RMSprop + momentum)

```py
HAI_NN.Optimizers.Adam(
    learning_rate=0.001,
    decay=0.,
    epsilon=1e-7,
    beta_1=0.9,
    beta_2=0.999
)
```

Args:

1. learning_rate: initial learning rate
2. decay: learning rate decay rate
3. epsilon: small floating point used to maintain numerical stability
4. beta_1: The exponential decay rate for the 1st moment estimates, similar to momentum argument in SGD
5. beta_2: The exponential decay rate for the 2st moment estimates, equivalent to rho argument in RMSprop

Update rule:

$$ momentums_{parameter} = beta_1 \times momentums_{parameter} - (1 - beta_1) \times gradient $$

$$ cache_{parameter} = beta_2 \times cache_{parameter} + (1 - beta_2) \times gradient^2 $$

- Note: bias correction mechanism is applied to the moving averages:

$$ momentums_{parameter,corrected} = {momentums_{parameter} \over {1 - beta_1^{iterations + 1}}} $$

$$ cache_{parameter,corrected} = {cache_{parameter} \over {1 - beta_2^{iterations + 1}}} $$

finally the parameter is updated:

$$ parameter = parameter - lr_{current} \times {momentums_{parameter,corrected} \over \sqrt{cache_{parameter,corrected}} + self.epsilon} $$

#

## Metrics.Accuracies

All accuracy metrics calculate the accuracy by obtaining the mean of the comparison vector between the predicted and ground-truth vectors

Average accuracy is then calculated for the whole batch

#

### Regression accuracy

Compares prediction to ground truth using following formula:

$$ |predictions - y| < precision $$

where precision is initialized using the training ground truth vector and the formula:

$$ precision = {std(y) \over 250} $$

#

### Categorical accuracy

Compares predicted class label to true label: 

$$ predictions == y $$

#

## Model

Groups layers into an object with training and inference features

```py
HAI_NN.Model.Model(
    layers=None
)
```

Args:

1. layers: optional list of layers to add to the model, added in the same order
   

A model must be compiled before being trained or evaluated, it will raise a ModelNotCompiledException exception otherwise. For example:

```py
# # Instantiate model
model = Model.Model([
    Layers.Dense(X.shape[1], 128),
    Activations.ReLU(),
    Layers.Dense(128, 256),
    Activations.ReLU(),
    Layers.Dense(256, 128),
    Activations.ReLU(),
    Layers.Dense(128, 10),
    Activations.Softmax()
])

model.compile(loss=Losses.CategoricalCrossEntropy(),
              optimizer=Optimizers.Adam(decay=1e-4),
              accuracy=Accuracies.Categorical_Accuracy()
)

model.train(X, train_labels, validation_data=(X_test, test_labels),
            epochs=10, batch_size=128, print_every=100)
```

Predict method can still be called even if the model is not compiled

```py
confidences = model.predict(image_data)
```
#

### Methods:

**save**

Save pickled Model to file

```py
save(
    path
)
```

Args:

1. path: string, PathLike, path to save pickled model

#

**load**

Load pickled Model from file, returns Model object.

```py
load(
    path
)
```

Args:

1. path: string, PathLike, path to load pickled model

#

**save_parameters**

Save pickled list of layer parameters

```py
save_parameters(
    path
)
```

**get_parameters**

Returns list of tuples where the first element is the array of weights and the second is the vector of biases of the layer.

```py
get_parameters()
```

#

**set_parameters**

Set layer parameters using list of tuples containing the weights and biases of each layer

```py
set_parameters(
    parameters
)
```

Args:

1. List of tuples where the first element is the array of weights and the second is the vector of biases of the layer.

#

**compile**

Configure the model for training

```py
compile(
    loss,
    optimizer,
    accuracy
)
```

Args:

1. loss: loss function, Losses.loss instance
2. optimizer: Optimizers.optimizer instance
3. accuracy: accuracy metric, Accuracies.accuracy instance

#

**evaluate**

Return the loss and accuracy values for the model in test mode

```py
evaluate(
    X_val,
    y_val,
    batch_size=None
)
```

Args:

1. X_val: validation dataset samples
2. y_val: validation samples labels
3. batch_size: number of samples per batch of computation

#

**train**

Trains the model for a fixed number of epochs

```py
train(
    X,
    y,
    epochs=1,
    batch_size=None,
    print_every=1,
    validation_data=None
)
```

Args:

1. X: training samples
2. y: training samples labels
3. epochs: number of training epochs
4. batch_size: number of samples per gradient update, if None then the whole training set is used
5. print_every: console output update rate, per training step
6. validation_data: tuple containing validation set samples and labels

#

**predict**

Generates output predictions for the input samples

```py
predict(
    X,
    batch_size=None
)
```

Args:

1. X: inputs samples
2. batch_size: number of samples per batch of computation

#

**history**

Returns a lists containing the loss and accuracy values over training epochs

```py
history()
```