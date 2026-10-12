# ============================================================
# TRANSFORMER V1 — PATCH OCCLUSION XAI
# ============================================================

import os
import numpy as np
import torch
from PIL import Image
from torchvision import transforms


# ============================================================
# PREPROCESSING
# ============================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


# ============================================================
# PATCH OCCLUSION
# ============================================================

def generate_patch_occlusion(model, image_path):

    image = Image.open(image_path).convert("RGB")

    input_tensor = transform(image).unsqueeze(0)

    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)

    model.eval()

    # --------------------------------------------------------
    # Original prediction
    # --------------------------------------------------------

    with torch.no_grad():
        original_output = model(input_tensor)

    original_output = original_output[0]

    original_sbp = original_output[0].item()
    original_dbp = original_output[1].item()

    # --------------------------------------------------------
    # Patch configuration
    # --------------------------------------------------------

    image_size = 224
    patch_size = 16

    num_patches_side = image_size // patch_size
    total_patches = num_patches_side ** 2

    # --------------------------------------------------------
    # Attribution maps
    # --------------------------------------------------------

    sbp_attribution = np.zeros(
        (num_patches_side, num_patches_side),
        dtype=np.float32
    )

    dbp_attribution = np.zeros(
        (num_patches_side, num_patches_side),
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Occlude each patch
    # --------------------------------------------------------

    for row in range(num_patches_side):

        for col in range(num_patches_side):

            occluded = input_tensor.clone()

            y1 = row * patch_size
            y2 = y1 + patch_size

            x1 = col * patch_size
            x2 = x1 + patch_size

            # Replace patch with zero
            occluded[
                :,
                :,
                y1:y2,
                x1:x2
            ] = 0.0

            with torch.no_grad():

                output = model(occluded)

            output = output[0]

            # Change caused by removing the patch
            sbp_change = abs(
                original_sbp - output[0].item()
            )

            dbp_change = abs(
                original_dbp - output[1].item()
            )

            sbp_attribution[row, col] = sbp_change
            dbp_attribution[row, col] = dbp_change

    # --------------------------------------------------------
    # Normalize maps
    # --------------------------------------------------------

    if sbp_attribution.max() > 0:

        sbp_attribution /= (
            sbp_attribution.max() + 1e-8
        )

    if dbp_attribution.max() > 0:

        dbp_attribution /= (
            dbp_attribution.max() + 1e-8
        )

    # --------------------------------------------------------
    # Return
    # --------------------------------------------------------

    return {
        "sbp_attribution": sbp_attribution,
        "dbp_attribution": dbp_attribution,
        "total_patches": total_patches,
        "patch_size": patch_size,
        "original_sbp": original_sbp,
        "original_dbp": original_dbp
    }