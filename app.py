# =========================
# IMPORTS
# =========================
import os
import gdown
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
from flask import Flask, render_template, request
from torchvision import models, transforms
from ultralytics import YOLO

# =========================
# PATHS (RELATIVE — CLOUD SAFE)
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
STATIC_DIR = os.path.join(BASE_DIR, "static")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

CLASSIFIER_PATH = os.path.join(MODEL_DIR, "clinicscan_classifier_15class.pth")
YOLO_PATH = os.path.join(MODEL_DIR, "best.pt")
OUTPUT_IMAGE_PATH = os.path.join(STATIC_DIR, "output.png")

# =========================
# GOOGLE DRIVE IDS
# =========================
CLASSIFIER_ID = "1SOaEcC9q29PL2ocrBLAg8QkMdxw-QjV2"
YOLO_ID = "1xzYVtQKGBvle7PPi4-XgwErhe7kSOiAm"

# =========================
# DOWNLOAD MODELS (RUNS ON STARTUP)
# =========================
def download_models():
    if not os.path.exists(CLASSIFIER_PATH):
        gdown.download(
            f"https://drive.google.com/uc?id={CLASSIFIER_ID}",
            CLASSIFIER_PATH,
            quiet=False
        )

    if not os.path.exists(YOLO_PATH):
        gdown.download(
            f"https://drive.google.com/uc?id={YOLO_ID}",
            YOLO_PATH,
            quiet=False
        )

download_models()

# =========================
# FLASK APP
# =========================
app = Flask(__name__)

# =========================
# LABELS & DEFINITIONS
# =========================
CLASS_NAMES = [
    "Aortic enlargement", "Atelectasis", "Calcification", "Cardiomegaly",
    "Consolidation", "ILD", "Infiltration", "Lung opacity",
    "Nodule", "Other lesion", "Pleural effusion",
    "Pleural thickening", "Pneumothorax",
    "Pulmonary fibrosis", "No finding"
]

DISEASE_DEFINITIONS = {
    "Aortic enlargement": "Enlargement of the aorta.",
    "Atelectasis": "Collapsed lung tissue.",
    "Calcification": "Calcium deposits in tissues.",
    "Cardiomegaly": "Enlarged heart.",
    "Consolidation": "Fluid-filled lung tissue.",
    "ILD": "Interstitial lung disease.",
    "Infiltration": "Inflammatory lung shadows.",
    "Lung opacity": "Diffuse hazy lung regions.",
    "Nodule": "Rounded lung lesion.",
    "Other lesion": "Unclassified abnormality.",
    "Pleural effusion": "Fluid around lungs.",
    "Pleural thickening": "Thickened pleural lining.",
    "Pneumothorax": "Collapsed lung due to air.",
    "Pulmonary fibrosis": "Lung scarring.",
    "No finding": "No abnormality detected."
}

# =========================
# LOAD MODELS (AFTER DOWNLOAD)
# =========================
@torch.no_grad()
def load_classifier():
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(2048, 15)
    model.load_state_dict(torch.load(CLASSIFIER_PATH, map_location="cpu"))
    model.eval()
    return model

classifier = load_classifier()
yolo_model = YOLO(YOLO_PATH)

# =========================
# PREPROCESS
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# =========================
# ROUTES
# =========================
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        image = Image.open(request.files["image"]).convert("RGB")

        # -------- YOLO Detection ONLY (classification hidden) --------
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        results = yolo_model(img_cv, conf=0.5, iou=0.4, max_det=5)[0]

        best_boxes = {}
        detections = []

        if results.boxes:
            for box in results.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                name = yolo_model.names[cls_id]

                if name not in best_boxes or conf > best_boxes[name]["conf"]:
                    best_boxes[name] = {
                        "box": box.xyxy[0].cpu().numpy().astype(int),
                        "conf": round(conf, 2)
                    }

        for disease, data in best_boxes.items():
            x1, y1, x2, y2 = data["box"]
            conf = data["conf"]

            cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                img_cv, f"{disease} {conf}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
            )

            detections.append((disease, conf))

        cv2.imwrite(OUTPUT_IMAGE_PATH, img_cv)

        definitions = {
            d: DISEASE_DEFINITIONS.get(d, "Definition not available.")
            for d, _ in detections
        }

        return render_template(
            "index.html",
            detections=detections,
            definitions=definitions,
            image_path="static/output.png"
        )

    return render_template("index.html")
