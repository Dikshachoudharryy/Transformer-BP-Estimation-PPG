import torch
from PIL import Image
from torchvision import transforms

from model import load_model, DEVICE


# ============================================================
# TEST IMAGE
# ============================================================

IMAGE_PATH = (
    r"C:\Users\diksh\OneDrive\Desktop\VT_Project"
    r"\BP_Dataset\test\images"
    r"\p11109975_82800131_82800131_0002_1.png"
)


# ============================================================
# LOAD MODEL
# ============================================================

model = load_model()


# ============================================================
# LOAD IMAGE
# ============================================================

image = Image.open(IMAGE_PATH).convert("RGB")

print()
print("Image size:", image.size)
print("Image mode:", image.mode)


# ============================================================
# PREPROCESS
# ============================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

image_tensor = transform(image)

image_tensor = image_tensor.unsqueeze(0)

print("Input shape:", image_tensor.shape)
print(
    "Input range:",
    image_tensor.min().item(),
    "to",
    image_tensor.max().item()
)


# ============================================================
# PREDICTION
# ============================================================

image_tensor = image_tensor.to(DEVICE)

with torch.no_grad():

    output = model(image_tensor)


# ============================================================
# NORMALIZED OUTPUT
# ============================================================

print()
print("Normalized output:")
print(output)


# ============================================================
# DENORMALIZE
# ============================================================

SBP_MEAN = 122.304856938613
SBP_STD = 21.051077265426

DBP_MEAN = 52.261745783263
DBP_STD = 11.949080997791


sbp = (
    output[0, 0].item() * SBP_STD
    + SBP_MEAN
)

dbp = (
    output[0, 1].item() * DBP_STD
    + DBP_MEAN
)


# ============================================================
# FINAL BP
# ============================================================

print()
print("=" * 60)
print("TRANSFORMER V1 BP PREDICTION")
print("=" * 60)

print(f"Predicted SBP : {sbp:.2f} mmHg")
print(f"Predicted DBP : {dbp:.2f} mmHg")

print("=" * 60)
print("REAL IMAGE INFERENCE TEST COMPLETE")
print("=" * 60)