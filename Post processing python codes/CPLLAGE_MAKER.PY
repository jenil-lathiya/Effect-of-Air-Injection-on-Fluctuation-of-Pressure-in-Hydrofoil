import os
import matplotlib.pyplot as plt
from PIL import Image

# --------- PATH TO YOUR IMAGES ---------
FOLDER = "last_folder/post-processing/q-criteria"

images = [
    "0.1-1.jpeg",
    "0.1-2.jpeg",
    "0.1-3.jpeg",
    "0.1-4.jpeg",
    "0.1-5.jpeg",
    "0.1-6.jpeg",
]

image_paths = [os.path.join(FOLDER, img) for img in images]

# --------- SIDE CROP FUNCTION ---------
def center_crop_width(img, target_width):
    """
    Crops only left & right sides.
    Height is preserved completely.
    """
    w, h = img.size

    if target_width >= w:
        return img  # nothing to crop

    left = (w - target_width) // 2
    right = left + target_width

    return img.crop((left, 0, right, h))

# --------- CHOOSE WIDTH (IMPORTANT) ---------
TARGET_WIDTH = 3000  # ← adjust if needed (1400–1600 typical)

# --------- CREATE COLLAGE ---------
fig, axes = plt.subplots(2, 3, figsize=(14, 7))

for ax, img_path in zip(axes.flatten(), image_paths):
    img = Image.open(img_path)

    # ✅ crop only sides, keep height
    img = center_crop_width(img, TARGET_WIDTH)

    ax.imshow(img)
    ax.axis("off")

plt.tight_layout()
plt.savefig("Q_criterion_collage.png", dpi=300, bbox_inches="tight")
plt.show()
