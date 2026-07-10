import NeuralNetwork
from Layers import Conv, ReLU, Pooling, Flatten, FullyConnected, SoftMax
from Optimization import Optimizers, Constraints
from Layers import Initializers


def build():
    optimizer = Optimizers.Adam(5e-4, 0.9, 0.999)
    optimizer.add_regularizer(Constraints.L2_Regularizer(4e-4))

    weights_initializer = Initializers.He()
    bias_initializer = Initializers.Constant(0.1)

    net = NeuralNetwork.NeuralNetwork(optimizer, weights_initializer, bias_initializer)

    # input: (batch, 1, 28, 28)
    net.append_layer(Conv.Conv(stride_shape=(1, 1), convolution_shape=(1, 5, 5), num_kernels=6))
    net.append_layer(ReLU.ReLU())
    net.append_layer(Pooling.Pooling(stride_shape=(2, 2), pooling_shape=(2, 2)))
    # -> (batch, 6, 14, 14)

    net.append_layer(Conv.Conv(stride_shape=(1, 1), convolution_shape=(6, 5, 5), num_kernels=16))
    net.append_layer(ReLU.ReLU())
    net.append_layer(Pooling.Pooling(stride_shape=(2, 2), pooling_shape=(2, 2)))
    # -> (batch, 16, 7, 7)

    net.append_layer(Flatten.Flatten())
    # -> (batch, 784)

    net.append_layer(FullyConnected.FullyConnected(784, 120))
    net.append_layer(ReLU.ReLU())

    net.append_layer(FullyConnected.FullyConnected(120, 84))
    net.append_layer(ReLU.ReLU())

    net.append_layer(FullyConnected.FullyConnected(84, 10))
    net.append_layer(SoftMax.SoftMax())

    return net
