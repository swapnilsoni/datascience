"""
'''This section demonstrates Singular Value Decomposition (SVD) for image compression.

First, a binary 8x8 image (`image_face`) is defined as a NumPy array.

SVD is then applied to `image_face` using `np.linalg.svd`, which decomposes the matrix into three components:
- `U`: The unitary matrix (left singular vectors)
- `S`: The singular values (as a 1D array)
- `Vh`: The unitary matrix (conjugate transpose of right singular vectors)

The shapes of these components are then displayed.

A dimensionality reduction is performed by selecting `k = 3` principal components (singular values and corresponding vectors).

Finally, the `rebuilded_image` is constructed using only these `k` components:
`rebuilded_image = U[:, : k] @ np.diag(S[: k]) @ Vh[: k, :]`

This reconstructed image is then displayed, showing the approximation of the original image using fewer components.
'''
```
"""

import numpy as np
import matplotlib.pyplot as plt

image_face = np.array([
    [0, 0, 1, 1, 1, 1, 0, 0],
    [0, 1, 0, 0, 0, 0, 1, 0],
    [1, 0, 1, 0, 0, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 0, 0, 1, 0, 1],
    [1, 0, 0, 1, 1, 0, 0, 1],
    [0, 1, 0, 0, 0, 0, 1, 0],
    [0, 0, 1, 1, 1, 1, 0, 0]
])

plt.imshow(image_face, cmap='gray_r')

U, S, Vh = np.linalg.svd(image_face)

U.shape, S.shape, Vh.shape

k = 3

U[:, : k].shape, S[: k].shape, Vh[: k, :].shape

rebuilded_image = U[:, : k] @ np.diag(S[: k]) @ Vh[: k, :]

plt.imshow(rebuilded_image, cmap='gray_r')
