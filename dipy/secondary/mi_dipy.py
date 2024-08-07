import imageio
import matplotlib.pyplot as plt
import numpy as np

from dipy.align.imaffine import (
    AffineRegistration,
    MutualInformationMetric,
    transform_centers_of_mass,
)
from dipy.align.transforms import (
    AffineTransform2D,
    RigidTransform2D,
    TranslationTransform2D,
)

# Load the static (fixed) image
static = imageio.imread("brain.png", mode="L").astype(np.float32)

# Load the moving image
moving = imageio.imread("distorted.png", mode="L").astype(np.float32)

# Ensure both images have the same shape
if static.shape != moving.shape:
    raise ValueError("The static and moving images must have the same dimensions.")

# Create affine matrices for the images
static_grid2world = np.eye(3)
moving_grid2world = np.eye(3)

# Create the Mutual Information metric
nbins = 32
sampling_prop = None
metric = MutualInformationMetric(nbins, sampling_prop)

# Create the affine registration object
affreg = AffineRegistration(
    metric=metric,
    level_iters=[10000, 1000, 100],
    sigmas=[3.0, 1.0, 0.0],
    factors=[4, 2, 1],
)

# Center of Mass Transform
c_of_mass = transform_centers_of_mass(
    static, static_grid2world, moving, moving_grid2world
)

# Translation Transform
transform = TranslationTransform2D()
params0 = None
starting_affine = c_of_mass.affine
translation = affreg.optimize(
    static,
    moving,
    transform,
    params0,
    static_grid2world,
    moving_grid2world,
    starting_affine=starting_affine,
)

# Rigid Transform
transform = RigidTransform2D()
params0 = None
starting_affine = translation.affine
rigid = affreg.optimize(
    static,
    moving,
    transform,
    params0,
    static_grid2world,
    moving_grid2world,
    starting_affine=starting_affine,
)

# Affine Transform
transform = AffineTransform2D()
params0 = None
starting_affine = rigid.affine
affine = affreg.optimize(
    static,
    moving,
    transform,
    params0,
    static_grid2world,
    moving_grid2world,
    starting_affine=starting_affine,
)

# Apply the transformation to the moving image
transformed = affine.transform(moving)


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
