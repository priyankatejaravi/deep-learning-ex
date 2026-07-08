# Utility helpers used for testing, gradients and datasets
import numpy as np
import matplotlib.pyplot as plt
import os
from random import shuffle
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import load_iris, load_digits


"""Collection of helper utilities for testing and dataset handling.

Includes numerical gradient checks, dataset wrappers (Iris/Digit), and simple
data shuffling/accuracy helpers used by the tests.
"""


# Function entry point
def compute_bn_gradients(error_tensor, input_tensor, weights, mean, var, eps=1e-15):
    """Compute gradient of batch normalization w.r.t. the input tensor.

    This uses the full chain-rule derivation through the normalization step.
    It does NOT compute gradients w.r.t. gamma/beta — do that separately.

    Args:
        error_tensor: gradient flowing in from the next layer (same shape as input)
        input_tensor: the original (unnormalized) input from the forward pass
        weights: gamma (scale) parameter of shape (channels,)
        mean: batch mean computed during forward (shape: channels,)
        var: batch variance computed during forward (shape: channels,)
        eps: small value for numerical stability
    Returns:
        gradient w.r.t. input_tensor
    """
    n = input_tensor.shape[0]  # batch size

    # Normalize just like we did in forward
    x_hat = (input_tensor - mean) / np.sqrt(var + eps)

    # Gradient w.r.t. x_hat from the error and gamma
    dx_hat = error_tensor * weights

    # Gradient w.r.t. variance
    dvar = np.sum(dx_hat * (input_tensor - mean) * -0.5 * (var + eps) ** (-1.5), axis=0)

    # Gradient w.r.t. mean
    dmean = np.sum(dx_hat * -1 / np.sqrt(var + eps), axis=0) + dvar * np.sum(-2 * (input_tensor - mean), axis=0) / n

    # Final gradient w.r.t. input
    dx = dx_hat / np.sqrt(var + eps) + dvar * 2 * (input_tensor - mean) / n + dmean / n
    return dx



# Function entry point
def gradient_check(layers, input_tensor, label_tensor):
    epsilon = 1e-5
    difference = np.zeros_like(input_tensor)

    activation_tensor = input_tensor.copy()
    for layer in layers[:-1]:
        activation_tensor = layer.forward(activation_tensor)
    layers[-1].forward(activation_tensor, label_tensor)

    error_tensor = layers[-1].backward(label_tensor)
    for layer in reversed(layers[:-1]):
        error_tensor = layer.backward(error_tensor)

    it = np.nditer(input_tensor, flags=['multi_index'])
    while not it.finished:
        plus_epsilon = input_tensor.copy()
        plus_epsilon[it.multi_index] += epsilon
        minus_epsilon = input_tensor.copy()
        minus_epsilon[it.multi_index] -= epsilon

        analytical_derivative = error_tensor[it.multi_index]

        for layer in layers[:-1]:
            plus_epsilon = layer.forward(plus_epsilon)
            minus_epsilon = layer.forward(minus_epsilon)
        upper_error = layers[-1].forward(plus_epsilon, label_tensor)
        lower_error = layers[-1].forward(minus_epsilon, label_tensor)

        numerical_derivative = (upper_error - lower_error) / (2 * epsilon)

        # print('Analytical: ' + str(analytical_derivative) + ' vs Numerical :' + str(numerical_derivative))
        normalizing_constant = max(np.abs(analytical_derivative), np.abs(numerical_derivative))

        if normalizing_constant < 1e-15:
            difference[it.multi_index] = 0
        else:
            difference[it.multi_index] = np.abs(analytical_derivative - numerical_derivative) / normalizing_constant

        it.iternext()
    return difference


# Function entry point
def gradient_check_weights(layers, input_tensor, label_tensor, bias):
    epsilon = 1e-5
    if bias:
        weights = layers[0].bias
    else:
        weights = layers[0].weights
    difference = np.zeros_like(weights)

    it = np.nditer(weights, flags=['multi_index'])
    while not it.finished:
        plus_epsilon = weights.copy()
        plus_epsilon[it.multi_index] += epsilon
        minus_epsilon = weights.copy()
        minus_epsilon[it.multi_index] -= epsilon

        activation_tensor = input_tensor.copy()
        if bias:
            layers[0].bias = weights
        else:
            layers[0].weights = weights
        for layer in layers[:-1]:
            activation_tensor = layer.forward(activation_tensor)
        layers[-1].forward(activation_tensor, label_tensor)

        error_tensor = layers[-1].backward(label_tensor)
        for layer in reversed(layers[:-1]):
            error_tensor = layer.backward(error_tensor)
        if bias:
            analytical_derivative = layers[0].gradient_bias
        else:
            analytical_derivative = layers[0].gradient_weights
        analytical_derivative = analytical_derivative[it.multi_index]

        if bias:
            layers[0].bias = plus_epsilon
        else:
            layers[0].weights = plus_epsilon
        plus_epsilon_activation = input_tensor.copy()
        for layer in layers[:-1]:
            plus_epsilon_activation = layer.forward(plus_epsilon_activation)

        if bias:
            layers[0].bias = minus_epsilon
        else:
            layers[0].weights = minus_epsilon
        minus_epsilon_activation = input_tensor.copy()
        for layer in layers[:-1]:
            minus_epsilon_activation = layer.forward(minus_epsilon_activation)

        upper_error = layers[-1].forward(plus_epsilon_activation, label_tensor)
        lower_error = layers[-1].forward(minus_epsilon_activation, label_tensor)

        numerical_derivative = (upper_error - lower_error) / (2 * epsilon)
        normalizing_constant = max(np.abs(analytical_derivative), np.abs(numerical_derivative))

        if normalizing_constant < 1e-15:
            difference[it.multi_index] = 0
        else:
            difference[it.multi_index] = np.abs(analytical_derivative - numerical_derivative) / normalizing_constant


        it.iternext()
    return difference


# Function entry point
def calculate_accuracy(results, labels):
    """Compute accuracy given soft predictions `results` and one-hot `labels`.

    Returns fraction of correctly predicted samples.
    """

    index_maximum = np.argmax(results, axis=1)
    one_hot_vector = np.zeros_like(results)
    for i in range(one_hot_vector.shape[0]):
        one_hot_vector[i, index_maximum[i]] = 1

    correct = 0.
    wrong = 0.
    for column_results, column_labels in zip(one_hot_vector, labels):
        if column_results[column_labels > 0.].all() > 0.:
            correct += 1.
        else:
            wrong += 1.

    return correct / (correct + wrong)


# Function entry point
def shuffle_data(input_tensor, label_tensor):
    """Shuffle inputs and labels in the same random order and return arrays."""
    index_shuffling = [i for i in range(input_tensor.shape[0])]
    shuffle(index_shuffling)
    shuffled_input = [input_tensor[i, :] for i in index_shuffling]
    shuffled_labels = [label_tensor[i, :] for i in index_shuffling]
    return (np.array(shuffled_input)), (np.array(shuffled_labels))



# Main component implementation
class RandomData:
    # Function entry point
    def __init__(self, input_size, batch_size, categories):
        self.input_size = input_size
        self.batch_size = batch_size
        self.categories = categories
        self.label_tensor = np.zeros([self.batch_size, self.categories])

    # Function entry point
    def next(self):
        input_tensor = np.random.random([self.batch_size, self.input_size])

        self.label_tensor = np.zeros([self.batch_size, self.categories])
        for i in range(self.batch_size):
            self.label_tensor[i, np.random.randint(0, self.categories)] = 1

        return input_tensor, self.label_tensor




# Main component implementation
class IrisData:
    # Function entry point
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self._data = load_iris()
        # sparse_output=False gives a dense array (sparse=False was deprecated in sklearn 1.2)
        self._label_tensor = OneHotEncoder(sparse_output=False).fit_transform(self._data.target.reshape(-1, 1))
        self._input_tensor = self._data.data
        self._input_tensor /= np.abs(self._input_tensor).max()

        self.split = int(self._input_tensor.shape[0]*(2/3))  # train / test split  == number of samples in train set

        self._input_tensor, self._label_tensor = shuffle_data(self._input_tensor, self._label_tensor)
        self._input_tensor_train = self._input_tensor[:self.split, :]
        self._label_tensor_train = self._label_tensor[:self.split, :]
        self._input_tensor_test = self._input_tensor[self.split:, :]
        self._label_tensor_test = self._label_tensor[self.split:, :]

        self._current_forward_idx_iterator = self._forward_idx_iterator()

    # Function entry point
    def _forward_idx_iterator(self):
        num_iterations = int(np.ceil(self.split / self.batch_size))
        idx = np.arange(self.split)
        while True:
            this_idx = np.random.choice(idx, self.split, replace=False)
            for i in range(num_iterations):
                yield this_idx[i * self.batch_size:(i + 1) * self.batch_size]

    # Function entry point
    def next(self):
        idx = next(self._current_forward_idx_iterator)
        return self._input_tensor_train[idx, :], self._label_tensor_train[idx, :]

    # Function entry point
    def get_test_set(self):
        return self._input_tensor_test, self._label_tensor_test



# Main component implementation
class DigitData:
    # Function entry point
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self._data = load_digits(n_class=10)
        # sparse_output=False gives a dense array (sparse=False was deprecated in sklearn 1.2)
        self._label_tensor = OneHotEncoder(sparse_output=False).fit_transform(self._data.target.reshape(-1, 1))
        self._input_tensor = self._data.data.reshape(-1, 1, 8, 8)
        self._input_tensor /= np.abs(self._input_tensor).max()

        self.split = int(self._input_tensor.shape[0]*(2/3))  # train / test split  == number of samples in train set

        self._input_tensor, self._label_tensor = shuffle_data(self._input_tensor, self._label_tensor)
        self._input_tensor_train = self._input_tensor[:self.split, :]
        self._label_tensor_train = self._label_tensor[:self.split, :]
        self._input_tensor_test = self._input_tensor[self.split:, :]
        self._label_tensor_test = self._label_tensor[self.split:, :]

        self._current_forward_idx_iterator = self._forward_idx_iterator()

    # Function entry point
    def _forward_idx_iterator(self):
        num_iterations = int(np.ceil(self.split / self.batch_size))
        rest = self.batch_size-self.split%self.batch_size
        idx = np.arange(self.split)
        while True:
            this_idx = np.random.choice(idx, self.split, replace=False)
            for i in range(num_iterations):
                if (i == num_iterations-1) and (rest != 0):
                    yield np.concatenate([this_idx[i * self.batch_size:(i + 1) * self.batch_size], this_idx[:rest]])
                else:
                    yield this_idx[i * self.batch_size:(i + 1) * self.batch_size]

    # Function entry point
    def next(self):
        idx = next(self._current_forward_idx_iterator)

        return self._input_tensor_train[idx, :], self._label_tensor_train[idx, :]

    # Function entry point
    def get_test_set(self):
        return self._input_tensor_test, self._label_tensor_test


# Main component implementation
class MNISTData:
    """Loads MNIST handwritten digit data for training LeNet.

    MNIST has 70,000 grayscale images of size 28x28.
    We split 60,000 for training and 10,000 for testing.
    Each image becomes shape (1, 28, 28) — one channel, 28 height, 28 width.
    Labels are one-hot encoded into 10 classes (digits 0-9).
    """

    # Function entry point
    def __init__(self, batch_size):
        self.batch_size = batch_size

        # Download MNIST via sklearn — cached locally after first run
        from sklearn.datasets import fetch_openml
        data = fetch_openml('mnist_784', version=1, as_frame=False)

        # Images are flat 784-dim vectors — reshape to (N, 1, 28, 28)
        images = data.data.reshape(-1, 1, 28, 28).astype(float)
        # Normalize pixel values from [0, 255] to [0, 1]
        images /= 255.0

        # Convert string labels like '0'..'9' to integers
        labels_int = data.target.astype(int)
        # One-hot encode: digit 3 becomes [0,0,0,1,0,0,0,0,0,0]
        labels_oh = OneHotEncoder(sparse_output=False).fit_transform(
            labels_int.reshape(-1, 1)
        )

        # Standard MNIST split: first 60k train, last 10k test
        self.split = 60000
        self._input_tensor_train = images[:self.split]
        self._label_tensor_train = labels_oh[:self.split]
        self._input_tensor_test = images[self.split:]
        self._label_tensor_test = labels_oh[self.split:]

        # Start the batch iterator
        self._current_forward_idx_iterator = self._forward_idx_iterator()

    # Function entry point
    def _forward_idx_iterator(self):
        """Yields random mini-batch indices forever (re-shuffles each epoch)."""
        num_iters = int(np.ceil(self.split / self.batch_size))
        idx = np.arange(self.split)
        while True:
            # Shuffle indices each epoch so batches are different
            shuffled = np.random.choice(idx, self.split, replace=False)
            for i in range(num_iters):
                yield shuffled[i * self.batch_size: (i + 1) * self.batch_size]

    # Function entry point
    def next(self):
        """Return the next mini-batch of (images, labels)."""
        idx = next(self._current_forward_idx_iterator)
        return self._input_tensor_train[idx], self._label_tensor_train[idx]

    # Function entry point
    def get_test_set(self):
        """Return the full test split for final evaluation."""
        return self._input_tensor_test, self._label_tensor_test

    # Function entry point
    def show_random_training_image(self):
        """Display one random training image — useful sanity check."""
        import matplotlib.pyplot as plt
        idx = np.random.randint(0, self.split)
        # squeeze removes the channel dimension for 2D plotting
        plt.figure('Random MNIST training sample')
        plt.imshow(self._input_tensor_train[idx, 0], cmap='gray')
        plt.title(f"Label: {np.argmax(self._label_tensor_train[idx])}")
        plt.show()
