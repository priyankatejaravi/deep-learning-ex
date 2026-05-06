import numpy as np
import matplotlib.pyplot as plt

class Checker:
    def __init__(self, resolution, tile_size):
        # resolution: total image size (width and height in pixels)
        # tile_size: side length of each black/white tile

        self.resolution = resolution
        self.tile_size = tile_size
        self.output = None

        if resolution % (2 * tile_size) != 0:
            raise ValueError("Resolution must be divisible by 2 * tile_size")

    def draw(self):
        # Start with a small 2x2 checker pattern
        base_pattern = np.array([[0, 1],
                                 [1, 0]])

        # Figure out how many times to repeat it across the image
        repeat_times = self.resolution // (2 * self.tile_size)
        pattern = np.tile(base_pattern, (repeat_times, repeat_times))

        # Scale each tile to the given tile size
        pattern = np.kron(pattern, np.ones((self.tile_size, self.tile_size)))

        # Save and return the generated image
        self.output = pattern
        return self.output.copy()

    def show(self):
        # Draw pattern if it hasn’t been generated yet
        if self.output is None:
            self.draw()

        # Display checkerboard as a grayscale image
        plt.imshow(self.output, cmap='gray')
        plt.axis('off')
        plt.show()      

class Circle:
    def __init__(self, resolution, radius, position):
        self.resolution = resolution
        self.radius = radius
        self.position = position
        self.output = None

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
        # resolution: image dimensions (square)
        self.resolution = resolution
        self.output = None

    def draw(self):
        # Create coordinate grids from 0 to 1
        x = np.linspace(0, 1, self.resolution)
        y = np.linspace(0, 1, self.resolution)
        xv, yv = np.meshgrid(x, y)

        # Define RGB channels:
        # Red → increases left to right
        # Green → increases top to bottom
        # Blue → decreases left to right
        r = xv
        g = yv
        b = 1 - xv

        # Stack channels to form a color image
        self.output = np.stack([r, g, b], axis=-1)
        return self.output.copy()

    def show(self):
        # Draw if image not yet created
        if self.output is None:
            self.draw()

        plt.imshow(self.output)
        plt.axis('off')
        plt.show()
