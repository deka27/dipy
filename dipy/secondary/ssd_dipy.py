import matplotlib.pyplot as plt
import numpy as np

from dipy.align.imaffine import AffineMap, AffineRegistration
from dipy.align.metrics import SSDMetric
from dipy.align.transforms import (
    AffineTransform2D,
    RigidTransform2D,
    TranslationTransform2D,
)

# Load the static (fixed) image
static = plt.imread("brain.png")[:, :, 0]

# Load the moving image
moving = plt.imread("distorted.png")[:, :, 0]

# Ensure both images have the same shape
if static.shape != moving.shape:
    raise ValueError("The static and moving images must have the same dimensions.")

# Normalize images
static = (static - static.min()) / (static.max() - static.min())
moving = (moving - moving.min()) / (moving.max() - moving.min())

# Create affine matrices for the images
static_grid2world = np.eye(3)
moving_grid2world = np.eye(3)

# Create the SSD metric
metric = SSDMetric(dim=2)

# Create the affine registration object
affreg = AffineRegistration(
    metric=metric,
    level_iters=[10000, 1000, 100],
    sigmas=[3.0, 1.0, 0.0],
    factors=[4, 2, 1],
)

# Translation Transform
transform = TranslationTransform2D()
params0 = None
translation = affreg.optimize(
    static, moving, transform, params0, static_grid2world, moving_grid2world
)

# Rigid Transform
transform = RigidTransform2D()
params0 = None
rigid = affreg.optimize(
    static,
    moving,
    transform,
    params0,
    static_grid2world,
    moving_grid2world,
    starting_affine=translation.affine,
)

# Affine Transform
transform = AffineTransform2D()
params0 = None
affine = affreg.optimize(
    static,
    moving,
    transform,
    params0,
    static_grid2world,
    moving_grid2world,
    starting_affine=rigid.affine,
)

# Apply the transformation to the moving image
affine_map = AffineMap(
    affine.affine, static.shape, static_grid2world, moving.shape, moving_grid2world
)
transformed = affine_map.transform(moving)


# Visualize results
def plot_images(image1, image2, image3, title1, title2, title3):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    ax1.imshow(image1, cmap="gray")
    ax1.set_title(title1)
    ax2.imshow(image2, cmap="gray")
    ax2.set_title(title2)
    ax3.imshow(image3, cmap="gray")
    ax3.set_title(title3)
    plt.show()


plot_images(
    static,
    moving,
    transformed,
    "Static Image",
    "Moving Image (Before)",
    "Transformed Moving Image (After)",
)

# Calculate final SSD
final_ssd = np.sum((static - transformed) ** 2)
print(f"Final Sum of Squared Differences: {final_ssd}")
