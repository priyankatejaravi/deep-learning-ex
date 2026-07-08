# LeNet architecture assembled from our existing framework layers
import sys
import os
# Make sure the parent folder (src_to_implement) is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import NeuralNetwork
from Layers import Conv, ReLU, Pooling, Flatten, FullyConnected, SoftMax, BatchNormalization
from Optimization import Optimizers, Constraints
from Layers import Initializers


# Function entry point
def build():
    """Build and return a LeNet-style CNN for MNIST (28x28 grayscale images).

    Architecture (simplified LeNet-5 variant):
        Input:  (batch, 1, 28, 28)

        Conv1:  6 kernels of size (1, 5, 5), stride (1,1)  → (batch, 6, 28, 28)
        ReLU
        Pool1:  window (2,2), stride (2,2)                  → (batch, 6, 14, 14)

        Conv2:  16 kernels of size (6, 5, 5), stride (1,1)  → (batch, 16, 14, 14)
        ReLU
        Pool2:  window (2,2), stride (2,2)                  → (batch, 16, 7, 7)

        Flatten                                              → (batch, 784)
        FC1:    784 → 120,  ReLU
        FC2:    120 → 84,   ReLU
        FC3:    84  → 10
        SoftMax                                              → (batch, 10)

    Optimizer: Adam(lr=5e-4, beta1=0.9, beta2=0.999)
    Regularizer: L2 with weight 4e-4 (applied to Adam)
    Weight init: He (good for ReLU networks)
    Bias init: Constant(0.1)
    """

    # -- Optimizer setup --------------------------------------------------
    # Adam with the learning rate from the description (5 * 10^-4)
    optimizer = Optimizers.Adam(learning_rate=5e-4, mu=0.9, rho=0.999)

    # L2 regularizer with weight 4 * 10^-4 as specified
    regularizer = Constraints.L2_Regularizer(alpha=4e-4)

    # Attach the regularizer to the optimizer so every layer that uses
    # this optimizer will automatically have L2 weight decay applied
    optimizer.add_regularizer(regularizer)

    # -- Initializers ------------------------------------------------------
    # He initialization is best for ReLU networks (prevents vanishing gradients)
    weights_init = Initializers.He()
    # Bias starts small and constant so it doesn't dominate at the start
    bias_init = Initializers.Constant(0.1)

    # -- Build the network -------------------------------------------------
    net = NeuralNetwork.NeuralNetwork(optimizer, weights_init, bias_init)

    # ---- Block 1: first convolution + pool ----
    # 6 filters of size 5x5 applied to the 1-channel input image
    # kernel_shape = (in_channels, height, width) = (1, 5, 5)
    net.append_layer(Conv.Conv(stride_shape=(1, 1),
                               convolution_shape=(1, 5, 5),
                               num_kernels=6))
    # Non-linearity after conv
    net.append_layer(ReLU.ReLU())
    # Halve the spatial size: 28x28 → 14x14
    net.append_layer(Pooling.Pooling(stride_shape=(2, 2), pooling_shape=(2, 2)))

    # ---- Block 2: second convolution + pool ----
    # 16 filters of size 5x5 applied to the 6-channel feature maps
    net.append_layer(Conv.Conv(stride_shape=(1, 1),
                               convolution_shape=(6, 5, 5),
                               num_kernels=16))
    net.append_layer(ReLU.ReLU())
    # Halve again: 14x14 → 7x7
    net.append_layer(Pooling.Pooling(stride_shape=(2, 2), pooling_shape=(2, 2)))

    # ---- Flatten and fully connected head ----
    # Collapse spatial dims: (batch, 16, 7, 7) → (batch, 784)
    net.append_layer(Flatten.Flatten())

    # FC1: 784 → 120
    net.append_layer(FullyConnected.FullyConnected(input_size=784, output_size=120))
    net.append_layer(ReLU.ReLU())

    # FC2: 120 → 84
    net.append_layer(FullyConnected.FullyConnected(input_size=120, output_size=84))
    net.append_layer(ReLU.ReLU())

    # FC3: 84 → 10 (one output per digit class)
    net.append_layer(FullyConnected.FullyConnected(input_size=84, output_size=10))

    # SoftMax turns raw scores into class probabilities (sums to 1)
    net.append_layer(SoftMax.SoftMax())

    return net
