import numpy as np
import cv2
import os
import urllib
import urllib.request
from zipfile import ZipFile

from HAI_NN import Model, Activations, Layers, Losses, Optimizers
from HAI_NN.Metrics import Accuracies


# # Data Preparation
# URL = 'https://nnfs.io/datasets/fashion_mnist_images.zip'
# FILE = 'fashion_mnist_images.zip'
# FOLDER = 'fashion_mnist_images'

# if not os.path.isfile(FILE):
#     print(f'Downloading {URL} and saving as {FILE}...')
#     urllib.request.urlretrieve(URL, FILE)

# print('Unzipping images...')
# with ZipFile(FILE) as zip_images:
#     zip_images.extractall(FOLDER)

# print('Done!')

def load_mnist_dataset(dataset, path):
    labels = os.listdir(os.path.join(path, dataset))
    
    X = []
    y = []
    
    # for each label folder
    for label in labels:
        for file in os.listdir(os.path.join(path, dataset, label)):
            # read image
            image = cv2.imread(os.path.join(path, dataset, label, file), cv2.IMREAD_UNCHANGED)
            
            X.append(image)
            y.append(label)
    
    return np.array(X), np.array(y).astype('uint8')

def create_data_mnist(path):
    X, y = load_mnist_dataset('train', path)
    X_test, y_test = load_mnist_dataset('test', path)
    
    return X, y, X_test, y_test

X, y, X_test, y_test = create_data_mnist('fashion_mnist_images')

# shuffle training set
# to avoid getting stuck in local minimum
keys = np.array(range(X.shape[0]))
np.random.shuffle(keys)
X = X[keys]
y = y[keys]

# flatten and scale features
X = (X.reshape(X.shape[0], -1).astype(np.float32) - 127.5) / 127.5
X_test = (X_test.reshape(X_test.shape[0], -1).astype(np.float32) - 127.5) / 127.5

# Instantiate model
model = Model.Model()

# add layers
model.add(Layers.Dense(X.shape[1], 128))
model.add(Activations.ReLU())
model.add(Layers.Dense(128, 128))
model.add(Activations.ReLU())
model.add(Layers.Dense(128, 10))
model.add(Activations.Softmax())

# set loss, optimizer, accuracy
model.compile(loss=Losses.CategoricalCrossEntropy(),
              optimizer=Optimizers.Adam(decay=1e-4),
              accuracy=Accuracies.Categorical_Accuracy()
)

model.finalize()

model.train(X, y, validation_data=(X_test, y_test),
            epochs=10, batch_size=128, print_every=100)

