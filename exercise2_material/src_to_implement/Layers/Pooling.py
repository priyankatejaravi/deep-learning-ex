import numpy as np
from Layers.Base import BaseLayer

class Pooling(BaseLayer):
    # constructor receives stride_shape and pooling_shape
    def __init__(self, stride_shape, pooling_shape):
        super().__init__()
        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape
        self._max_locations = None  # stores max positions for backward
        self._input_shape = None

    def forward(self, input_tensor):
        self._input_shape = input_tensor.shape
        B, C, H, W = input_tensor.shape
        ph, pw = self.pooling_shape
        sy, sx = self.stride_shape
        # valid padding: no padding, discard border elements
        out_H = (H - ph) // sy + 1
        out_W = (W - pw) // sx + 1

        output = np.zeros((B, C, out_H, out_W))
        self._max_locations = np.zeros((B, C, out_H, out_W, 2), dtype=int)

        for b in range(B):
            for c in range(C):
                for i in range(out_H):
                    for j in range(out_W):
                        y0, x0 = i * sy, j * sx
                        region = input_tensor[b, c, y0:y0+ph, x0:x0+pw]
                        output[b, c, i, j] = np.max(region)
                        # store position of max for backward pass
                        my, mx = np.unravel_index(np.argmax(region), region.shape)
                        self._max_locations[b, c, i, j] = [y0 + my, x0 + mx]

        return output

    def backward(self, error_tensor):
        error_input = np.zeros(self._input_shape)
        B, C, _, _ = self._input_shape
        _, _, out_H, out_W = error_tensor.shape

        for b in range(B):
            for c in range(C):
                for i in range(out_H):
                    for j in range(out_W):
                        y, x = self._max_locations[b, c, i, j]
                        # write error back to the position of the max
                        error_input[b, c, y, x] += error_tensor[b, c, i, j]

        return error_input
