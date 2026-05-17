class Sgd:
    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate

    # according to gradient descent update scheme: w(t+1) = w(t) - (Learning_rat * gradient)
    def calculate_update(self, weight_tensor, gradient_tensor):
        return weight_tensor - (self.learning_rate * gradient_tensor)