from HAI_NN import Layers, Optimizers, Losses, Model, Activations
from HAI_NN.Metrics import Accuracies
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

# Create train and test dataset
# Create dataset
X, y = spiral_data(samples=1000, classes=3)
X_test, y_test = spiral_data(samples=100, classes=3)
# Instantiate the model
model = Model.Model()
# Add layers
model.add(Layers.Dense(2, 512, weight_regularizer_L2=5e-4, bias_regularizer_L2=5e-4))
model.add(Activations.ReLU())
model.add(Layers.Dropout(0.1))
model.add(Layers.Dense(512, 3))
model.add(Activations.Softmax())
# Set loss, optimizer and accuracy objects
model.compile(
        loss=Losses.CategoricalCrossEntropy(),
        optimizer=Optimizers.Adam(learning_rate=0.05, decay=5e-5),
        accuracy=Accuracies.Categorical_Accuracy()
        )
# Finalize the model
model.finalize()
# Train the model
model.train(X, y, validation_data=(X_test, y_test), epochs=10000, print_every=100)
