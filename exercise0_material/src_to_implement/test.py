import numpy as np
import matplotlib.pyplot as plt


class Checker:
    """
    Generates a square checkerboard pattern.

    Parameters
    ----------
    resolution : int
        Side length of the square output in pixels.
        Must be divisible by ``2 * tile_size``.
    tile_size : int
        Side length (in pixels) of each individual black or white tile.
    """

    def __init__(self, resolution: int, tile_size: int):
        if resolution % (2 * tile_size) != 0:
            raise ValueError(
                f"resolution ({resolution}) must be divisible by "
                f"2 * tile_size ({2 * tile_size})."
            )
        self.resolution = resolution
        self.tile_size  = tile_size
        self.output: np.ndarray | None = None

    def draw(self) -> np.ndarray:
        """
        Build and store the checkerboard.

        Returns a *copy* of the internal array so callers cannot mutate state.
        """
        # Use np.indices to decide the parity of each pixel's tile column/row
        row_idx, col_idx = np.indices((self.resolution, self.resolution))
        tile_row = row_idx // self.tile_size
        tile_col = col_idx // self.tile_size

        # XOR of parities: 0 → black tile, 1 → white tile
        self.output = ((tile_row + tile_col) % 2).astype(np.uint8)
        return self.output.copy()

    def show(self):
        """Display the pattern (draws first if needed)."""
        if self.output is None:
            self.draw()
        plt.figure(figsize=(5, 5))
        plt.imshow(self.output, cmap='gray', vmin=0, vmax=1)
        plt.title("Checkerboard")
        plt.axis('off')
        plt.tight_layout()
        plt.show()


class Circle:
    """
    Renders a filled white circle on a black square canvas.

    Parameters
    ----------
    resolution : int
        Side length of the square canvas in pixels.
    radius : float
        Radius of the circle in pixels.
    position : tuple[int, int]
        (x, y) centre of the circle in pixel coordinates
        (x → horizontal, y → vertical).
    """

    def __init__(self, resolution: int, radius: float, position: tuple):
        self.resolution = resolution
        self.radius     = radius
        self.cx, self.cy = position          # (x, y) == (col, row)
        self.output: np.ndarray | None = None

    def draw(self) -> np.ndarray:
        """
        Rasterise the circle via a per-pixel distance test.

        Returns a *copy* of the boolean mask cast to uint8.
        """
        cols, rows = np.meshgrid(
            np.arange(self.resolution),
            np.arange(self.resolution),
        )
        dist_sq = (cols - self.cx) ** 2 + (rows - self.cy) ** 2
        self.output = (dist_sq <= self.radius ** 2).astype(np.uint8)
        return self.output.copy()

    def show(self):
        """Display the circle (draws first if needed)."""
        if self.output is None:
            self.draw()
        plt.figure(figsize=(5, 5))
        plt.imshow(self.output, cmap='gray', vmin=0, vmax=1)
        plt.title("Circle")
        plt.axis('off')
        plt.tight_layout()
        plt.show()


class Spectrum:
    """
    Generates a smooth RGB colour-spectrum image.

    Colour mapping
    --------------
    * Red   channel – increases left → right   (0 … 1)
    * Green channel – increases top  → bottom  (0 … 1)
    * Blue  channel – decreases left → right   (1 … 0)

    Parameters
    ----------
    resolution : int
        Side length of the square output in pixels.
    """

    def __init__(self, resolution: int):
        self.resolution = resolution
        self.output: np.ndarray | None = None

    def draw(self) -> np.ndarray:
        """
        Build the H×W×3 float64 array in [0, 1]³.

        Returns a *copy* of the internal array.
        """
        h = w = self.resolution

        # Horizontal gradient: 0 on left edge, 1 on right edge
        horiz = np.linspace(0.0, 1.0, w)           # shape (W,)
        # Vertical gradient:   0 on top  edge, 1 on bottom edge
        vert  = np.linspace(0.0, 1.0, h)[:, None]  # shape (H, 1)

        red   = np.broadcast_to(horiz,        (h, w)).copy()
        green = np.broadcast_to(vert,         (h, w)).copy()
        blue  = np.broadcast_to(1.0 - horiz,  (h, w)).copy()

        self.output = np.stack([red, green, blue], axis=-1)  # (H, W, 3)
        return self.output.copy()

    def show(self):
        """Display the spectrum (draws first if needed)."""
        if self.output is None:
            self.draw()
        plt.figure(figsize=(5, 5))
        plt.imshow(self.output)
        plt.title("Colour Spectrum")
        plt.axis('off')
        plt.tight_layout()
        plt.show()