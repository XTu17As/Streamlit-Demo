import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
import gdown
import os
import psutil
import time

from tinyvit_fcos import TinyViT_FCOS, postprocess_predictions

# =========================================================
# CONFIG
# =========================================================
MODEL_PATH = "fcos_tinyvit_checkpoint.pth"
DRIVE_FILE_ID = "16922btuT2tHbo-oEPR3Rl2xrq4xZMnyD"
DOWNLOAD_URL = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"

# =========================================================
# MODEL LOADING (cached)
# =========================================================
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model from Google Drive..."):
            gdown.download(DOWNLOAD_URL, MODEL_PATH, quiet=False)

    model = TinyViT_FCOS(num_classes=12)
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint.get("model", checkpoint), strict=False)
    model.eval()
    return model

model = load_model()

# =========================================================
# RESOURCE MONITOR
# =========================================================
def get_resources():
    cpu = psutil.cpu_percent()
    ram = psutil.virtual_memory().percent
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.memory_allocated() / 1024**2
        gpu_total = torch.cuda.get_device_properties(0).total_memory / 1024**2
        gpu_percent = gpu_mem / gpu_total * 100
        gpu_str = f"{gpu_mem:.1f}/{gpu_total:.1f} MB ({gpu_percent:.1f}%)"
    else:
        gpu_str = "GPU not available"
    return cpu, ram, gpu_str

# =========================================================
# APP UI
# =========================================================
st.title("🧠 TinyViT + FCOS Shelf Detection Demo with Resource Monitor")
st.write("Upload a retail shelf image to test detection.")

uploaded = st.file_uploader("📸 Upload Image", type=["jpg", "jpeg", "png"])

# Placeholders for dynamic updating
cpu_text = st.empty()
ram_text = st.empty()
gpu_text = st.empty()
progress_text = st.empty()

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    img_np = np.array(img)
    img_resized = cv2.resize(img_np, (512, 512))
    st.image(img_resized, caption="Uploaded Image", channels="RGB")

    tensor = torch.from_numpy(img_resized.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0

    st.info("Running inference...")

    # Monitor resources while inference runs
    with torch.no_grad():
        for i in range(5):  # simple loop to simulate monitoring during process
            cpu, ram, gpu_str = get_resources()
            cpu_text.text(f"CPU Usage: {cpu:.1f}%")
            ram_text.text(f"RAM Usage: {ram:.1f}%")
            gpu_text.text(f"GPU Usage: {gpu_str}")
            time.sleep(0.2)  # update interval

        # Run the actual inference
        cls_logits, reg_preds, ctrness = model(tensor)
        results = postprocess_predictions(cls_logits, reg_preds, ctrness, (512, 512), torch.device("cpu"))[0]

    # Visualize detections
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

    # Final resource stats
    cpu, ram, gpu_str = get_resources()
    st.write(f"**Final Resource Usage:** CPU {cpu:.1f}%, RAM {ram:.1f}%, GPU {gpu_str}")
