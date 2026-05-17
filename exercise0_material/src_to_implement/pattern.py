import numpy as np
import matplotlib.pyplot as plt

class Checker:
    def __init__(self, resolution, tile_size):
        # resolution: total image size (width and height in pixels)
        # tile_size: side length of each black/white tile
        self.resolution = resolution
        self.tile_size = tile_size
        if resolution % (2 * tile_size) != 0:
            raise ValueError(
                f"resolution ({resolution}) must be divisible by "
                f"2 * tile_size ({2 * tile_size})."
            )
        self.output: np.ndarray | None = None

    def draw(self):
        # Use np.indices to decide the parity of each pixel's tile column/row
        row_idx, col_idx = np.indices((self.resolution, self.resolution))
        tile_row = row_idx // self.tile_size
        tile_col = col_idx // self.tile_size

        # XOR of parities: 0 → black tile, 1 → white tile
        self.output = ((tile_row + tile_col) % 2).astype(np.uint8)
        return self.output.copy()

    def show(self):
        # Display the pattern
        if self.output is None:
            self.draw()
        plt.figure(figsize=(5, 5))
        plt.imshow(self.output, cmap='gray', vmin=0, vmax=1)
        plt.title("Checkerboard")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

class Circle:
    def __init__(self, resolution, radius, position):
        self.resolution = resolution
        self.radius = radius
        self.position = position
        self.output: np.ndarray | None = None

    def draw(self):
        x = np.arange(self.resolution)
        xv, yv = np.meshgrid(x, x)
        self.output = (xv - self.position[0])**2 + (yv - self.position[1])**2 <= self.radius**2
        return self.output.copy()

    def show(self):
        if self.output is None:
            self.draw()
        plt.imshow(self.output, cmap='gray')
        plt.axis('off')
        plt.show()

class Spectrum:
    def __init__(self, resolution):
        self.resolution = resolution
        self.output: np.ndarray | None = None

    def draw(self):
        spectrum_array = np.zeros([self.resolution, self.resolution, 3])
        spectrum_array[:, :, 0] = np.linspace(0, 1, self.resolution)
        spectrum_array[:, :, 1] = np.linspace(0, 1, self.resolution).reshape(self.resolution, 1)
        spectrum_array[:, :, 2] = np.linspace(1, 0, self.resolution)

        self.output = spectrum_array.copy()
        return spectrum_array.copy()

    def show(self):
        if self.output is None:
            self.draw()
        plt.imshow(self.output)
        plt.show()
