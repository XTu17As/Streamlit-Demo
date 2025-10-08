import streamlit as st
import torch
import gdown
import os
from tinyvit_fcos import TinyViT_FCOS

MODEL_PATH = "fcos_tinyvit_checkpoint.pth"
DRIVE_FILE_ID = "16922btuT2tHbo-oEPR3Rl2xrq4xZMnyD"  # your model ID

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model from Google Drive..."):
            url = f"https://drive.google.com/uc?export=download&id={DRIVE_FILE_ID}&confirm=1"
            gdown.download(url, MODEL_PATH, quiet=False, fuzzy=True)

    # Double-check file validity
    if os.path.getsize(MODEL_PATH) < 5_000_000:  # too small, likely HTML
        st.error("⚠️ Model download failed (got an HTML file instead). Please check sharing link.")
        raise RuntimeError("Downloaded file is not a valid model checkpoint")

    checkpoint = torch.load(MODEL_PATH, map_location="cpu")

    model = TinyViT_FCOS(num_classes=1)
    # Your checkpoint likely has {'model': state_dict, ...}
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        checkpoint = checkpoint["model"]
    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    return model
