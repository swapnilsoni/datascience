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

# 1. The image is just an 8x8 matrix of 0s and 1s
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

# 2. Run SVD:  image_face = U @ diag(S) @ Vh
U, S, Vh = np.linalg.svd(image_face)
print("U shape :", U.shape)     # (8, 8)
print("S shape :", S.shape)     # (8,)
print("Vh shape:", Vh.shape)    # (8, 8)
print("Singular values:", S.round(2))

# 3. Rebuild using only the top k "stretches"
k = 3
rebuilt_image = U[:, :k] @ np.diag(S[:k]) @ Vh[:k, :]

# 4. Show original vs rebuilt, side by side
fig, axes = plt.subplots(1, 2, figsize=(8, 4))
axes[0].imshow(image_face, cmap='gray_r')
axes[0].set_title("Original (full matrix)")
axes[0].axis('off')

axes[1].imshow(rebuilt_image, cmap='gray_r')
axes[1].set_title(f"Rebuilt from top {k} stretches")
axes[1].axis('off')

plt.tight_layout()
plt.show()
