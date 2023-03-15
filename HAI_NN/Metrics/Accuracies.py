
import numpy as np
# Base accuracy class
class Accuracy:
    def calculate(self, predictions, y):
        comparisons = self.compare(predictions, y)
        accuracy = np.mean(comparisons)
        return accuracy
    
class Regression_Accuracy(Accuracy):
    def __init__(self) -> None:
        self.precision = None
    
    def init(self, y, reinit=False):
        if self.precision is None or reinit:
            self.precision = np.std(y)/250
    
    def compare(self, predictions, y):
        return np.absolute(predictions - y) < self.precision

class Categorical_Accuracy(Accuracy):
    def init(self, y):
        pass

    def compare(self, predictions, y):
        if len(y.shape) == 2:
            y = np.argmax(y, axis = 1)
        return predictions == y