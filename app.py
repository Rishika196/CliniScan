import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "0"
os.environ["ULTRALYTICS_SETTINGS"] = "False"

import streamlit as st
import torch
import numpy as np
from PIL import Image, ImageDraw
from torchvision import transforms
from ultralytics import YOLO

# -----------------------------
# CONFIG
# -----------------------------
CLASSIFICATION_MODEL_PATH = "clinicscan_classifier_15class.pth"
YOLO_MODEL_PATH = "runs/detect/train38/weights/best.pt"

CLASS_NAMES = [
    "Aortic enlargement", "Atelectasis", "Calcification", "Cardiomegaly",
    "Consolidation", "ILD", "Infiltration", "Lung Opacity",
    "Nodule/Mass", "Other lesion", "Pleural effusion",
    "Pleural thickening", "Pneumothorax", "Pulmonary fibrosis",
    "No finding"
]

DESCRIPTION_MAP = {
    "Aortic enlargement": "Enlargement of the aorta suggesting vascular abnormality.",
    "Atelectasis": "Partial or complete collapse of lung tissue.",
    "Calcification": "Calcium deposits in lung tissue.",
    "Cardiomegaly": "Enlarged heart size.",
    "Consolidation": "Fluid-filled lung regions.",
    "ILD": "Interstitial lung disease affecting lung structure.",
    "Infiltration": "Abnormal substance accumulation in lungs.",
    "Lung Opacity": "Hazy areas possibly indicating infection or edema.",
    "Nodule/Mass": "Abnormal growth in lung tissue.",
    "Other lesion": "Other abnormal lung lesion.",
    "Pleural effusion": "Fluid accumulation around the lungs.",
    "Pleural thickening": "Thickening of pleural lining.",
    "Pneumothorax": "Air in pleural space causing lung collapse.",
    "Pulmonary fibrosis": "Scarring of lung tissue.",
    "No finding": "No abnormality detected."
}

DEVICE = "cpu"

# -----------------------------
# STREAMLIT PAGE SETUP
# -----------------------------
st.set_page_config(page_title="CliniScan", layout="wide")
st.title("🩺 CliniScan – Chest X-ray Analysis")

# -----------------------------
# LOAD MODELS (CACHED)
# -----------------------------
@st.cache_resource
def load_classifier():
    model = torch.load(CLASSIFICATION_MODEL_PATH, map_location=DEVICE)
    model.eval()
    return model

@st.cache_resource
def load_yolo():
    return YOLO(YOLO_MODEL_PATH)

classifier = load_classifier()
yolo_model = load_yolo()

# -----------------------------
# IMAGE PREPROCESS
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def classify_image(image):
    img = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = classifier(img)
        pred = torch.argmax(outputs, dim=1).item()
    return CLASS_NAMES[pred]

# -----------------------------
# PIL BOUNDING BOX DRAWING
# -----------------------------
def draw_boxes_pil(image, detections, names):
    draw = ImageDraw.Draw(image)
    for box in detections:
        x1, y1, x2, y2 = box["bbox"]
        label = f"{box['label']} {box['conf']:.2f}"
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1, max(0, y1 - 15)), label, fill="red")
    return image

# -----------------------------
# UI
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload Chest X-ray", type=["png", "jpg", "jpeg"]
)

zoom = st.slider("Zoom Image (%)", 50, 150, 80)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    # Resize for zoom
    w, h = image.size
    image_display = image.resize(
        (int(w * zoom / 100), int(h * zoom / 100))
    )

    st.image(image_display, caption="Uploaded X-ray", use_container_width=True)

    # -----------------------------
    # CLASSIFICATION
    # -----------------------------
    prediction = classify_image(image)
    st.subheader("🔍 Classification Result")
    st.success(f"Prediction: {prediction}")
    st.write(f"**Description:** {DESCRIPTION_MAP[prediction]}")

    # -----------------------------
    # YOLO DETECTION
    # -----------------------------
    st.subheader("📦 YOLO Detection (Primary)")

    results = yolo_model(np.array(image))

    detections = []
    detected_labels = set()

    for r in results:
        if r.boxes is not None:
            for b in r.boxes:
                cls_id = int(b.cls[0])
                label = yolo_model.names[cls_id]
                conf = float(b.conf[0])

                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())

                detections.append({
                    "bbox": (x1, y1, x2, y2),
                    "label": label,
                    "conf": conf
                })

                detected_labels.add(label)

    if detected_labels:
        st.success(f"Detected: {', '.join(detected_labels)}")
    else:
        st.warning("No abnormalities detected by YOLO.")

    # Draw boxes
    if detections:
        boxed_image = draw_boxes_pil(image.copy(), detections, yolo_model.names)
        st.image(
            boxed_image.resize(
                (int(w * zoom / 100), int(h * zoom / 100))
            ),
            caption="Detected Abnormalities",
            use_container_width=True
        )

    # -----------------------------
    # DISEASE DESCRIPTIONS
    # -----------------------------
    st.subheader("📖 Detected Abnormality Details")
    for label in detected_labels:
        st.write(f"**{label}:** {DESCRIPTION_MAP.get(label, 'No description available.')}")
