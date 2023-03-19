
import numpy as np
# Base accuracy class
class Accuracy:
    def compare(self, predictions: np.ndarray, y: np.ndarray):
        pass
    
    def calculate(self, predictions: np.ndarray, y: np.ndarray):
        comparisons = self.compare(predictions, y)
        accuracy = np.mean(comparisons)
        
        # add accumulated sum of matching values and sample count
        self.accumulated_sum += np.sum(comparisons)
        self.accumulated_count += len(comparisons)
        
        return accuracy
    
    def calculate_accumulated(self):
        accuracy = self.accumulated_sum / self.accumulated_count
        return accuracy
    
    # reset variables
    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0
    
class Regression_Accuracy(Accuracy):
    """
    A custom made accuracy metric
    """
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

    def compare(self, predictions, y: np.ndarray):
        if len(y.shape) == 2:
            y = np.argmax(y, axis = 1)
        return predictions == y