class Sgd:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def calculate_update(self, weights, gradients):
        # Update weights: w = w - learning_rate * gradient
        return weights - self.learning_rate * gradients