# ============================================================
# V1 TRANSFORMER MODEL LOADER
# ============================================================

import os
import json
import torch

from models.transformer_v1.model import BPTransformer


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models",
    "transformer_v1"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "transformer_v1_final.pth"
)

CONFIG_PATH = os.path.join(
    MODEL_DIR,
    "transformer_v1_config.json"
)


# ------------------------------------------------------------
# Device
# ------------------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------
# Load V1 Transformer
# ------------------------------------------------------------

def load_model():

    print("=" * 60)
    print("LOADING TRANSFORMER V1")
    print("=" * 60)

    print("Device:", DEVICE)
    print("Model:", MODEL_PATH)
    print("Config:", CONFIG_PATH)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"V1 model not found:\n{MODEL_PATH}"
        )

    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(
            f"V1 config not found:\n{CONFIG_PATH}"
        )

    # --------------------------------------------------------
    # Load configuration
    # --------------------------------------------------------

    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)

    print("\nV1 Configuration:")
    print(config)

    # --------------------------------------------------------
    # Create architecture
    # --------------------------------------------------------

    model = BPTransformer(
        image_size=config["image_size"],
        patch_size=config["patch_size"],
        in_channels=config["in_channels"],
        embed_dim=config["embed_dim"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        dropout=config["dropout"]
    )

    # --------------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------------

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    # --------------------------------------------------------
    # Handle checkpoint formats
    # --------------------------------------------------------

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:

            state_dict = checkpoint["model_state_dict"]

        elif "state_dict" in checkpoint:

            state_dict = checkpoint["state_dict"]

        else:

            # checkpoint itself may be state_dict
            state_dict = checkpoint

    else:

        state_dict = checkpoint

    model.load_state_dict(
        state_dict,
        strict=True
    )

    model = model.to(DEVICE)
    model.eval()

    print("\n✓ Transformer V1 loaded successfully")
    print("=" * 60)

    return model


# ------------------------------------------------------------
# Normalization values
# ------------------------------------------------------------

SBP_MEAN = 122.30485693861299
SBP_STD = 21.051077265426123

DBP_MEAN = 52.26174578326336
DBP_STD = 11.949080997790393


# ------------------------------------------------------------
# Prediction
# ------------------------------------------------------------

def predict(model, image_tensor):

    image_tensor = image_tensor.to(DEVICE)

    with torch.no_grad():

        output = model(image_tensor)

    output = output.cpu().numpy()

    sbp_normalized = output[:, 0]
    dbp_normalized = output[:, 1]

    sbp = (
        sbp_normalized * SBP_STD
        + SBP_MEAN
    )

    dbp = (
        dbp_normalized * DBP_STD
        + DBP_MEAN
    )

    return sbp, dbp