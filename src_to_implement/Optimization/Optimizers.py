class Sgd:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        # Update weights: w = w - learning_rate * gradient
        return weight_tensor - self.learning_rate * gradient_tensor 