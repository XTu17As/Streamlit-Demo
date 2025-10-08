import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
import gdown
import os

from tinyvit_fcos import TinyViT_FCOS, postprocess_predictions

# =========================================================
# CONFIG
# =========================================================
MODEL_PATH = "fcos_tinyvit_checkpoint.pth"
DRIVE_FILE_ID = "16922btuT2tHbo-oEPR3Rl2xrq4xZMnyD"  # your model file ID
DOWNLOAD_URL = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}&confirm=1"

# =========================================================
# MODEL LOADING (cached)
# =========================================================
@st.cache_resource
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model from Google Drive..."):
            url = f"https://drive.google.com/uc?export=download&id={DRIVE_FILE_ID}&confirm=1"
            gdown.download(url, MODEL_PATH, quiet=False, fuzzy=True)

    if os.path.getsize(MODEL_PATH) < 5_000_000:
        st.error("⚠️ Model file seems invalid (too small or HTML).")
        raise RuntimeError("Model download failed.")

    # --- Safe loading for full checkpoint ---
    try:
        checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(MODEL_PATH, map_location="cpu")

    model = TinyViT_FCOS(num_classes=1)

    # --- Extract state dict properly ---
    state_dict = checkpoint.get("model", checkpoint)

    # --- Ignore mismatch layers (like head.cls_logits) ---
    model_state = model.state_dict()
    filtered_state = {k: v for k, v in state_dict.items() if k in model_state and v.shape == model_state[k].shape}

    missing_keys = [k for k in model_state.keys() if k not in filtered_state]
    if missing_keys:
        st.warning(f"Ignoring {len(missing_keys)} mismatched layers (e.g., classification head).")

    model.load_state_dict(filtered_state, strict=False)
    model.eval()
    return model


# =========================================================
# APP UI
# =========================================================
st.title("🧠 TinyViT + FCOS Shelf Detection Demo")
st.write("Upload a retail shelf image to test detection.")

uploaded = st.file_uploader("📸 Upload Image", type=["jpg", "jpeg", "png"])

if uploaded:
    # Read image
    img = Image.open(uploaded).convert("RGB")
    img_np = np.array(img)
    img_resized = cv2.resize(img_np, (512, 512))
    st.image(img_resized, caption="Uploaded Image", channels="RGB")

    # Preprocess
    tensor = torch.from_numpy(img_resized.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0

    # Run inference
    model = load_model()
    st.info("Running inference...")
    with torch.no_grad():
        cls_logits, reg_preds, ctrness = model(tensor)
        results = postprocess_predictions(
            cls_logits, reg_preds, ctrness, (512, 512), torch.device("cpu")
        )[0]

    # Visualize
    for box, score, cls_id in results:
        x0, y0, x1, y1 = map(int, box)
        cv2.rectangle(img_resized, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2.putText(
            img_resized,
            f"{score:.2f}",
            (x0, max(15, y0 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )

    st.success("✅ Inference complete!")
    st.image(img_resized, caption="Detection Result", channels="RGB", use_container_width=True)
