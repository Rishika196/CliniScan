from flask import Flask, render_template, request
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import cv2
import os

from torchvision import models, transforms
from ultralytics import YOLO

app = Flask(__name__)

# =========================
# CONFIG
# =========================
CLASSIFICATION_MODEL_PATH = "models/clinicscan_classifier_15class.pth"
YOLO_MODEL_PATH = "models/yolo_best.pt"
OUTPUT_IMAGE_PATH = "static/output.png"

CLASS_NAMES = [
    "Aortic enlargement", "Atelectasis", "Calcification", "Cardiomegaly",
    "Consolidation", "ILD", "Infiltration", "Lung opacity",
    "Nodule", "Other lesion", "Pleural effusion",
    "Pleural thickening", "Pneumothorax",
    "Pulmonary fibrosis", "No finding"
]

DISEASE_DEFINITIONS = {
    "Aortic enlargement":"Enlargement of the aorta.",
    "Atelectasis":"Collapsed lung tissue.",
    "Calcification":"Calcium deposits in tissues.",
    "Cardiomegaly":"Enlarged heart.",
    "Consolidation":"Fluid-filled lung tissue.",
    "ILD":"Interstitial lung disease.",
    "Infiltration":"Inflammatory lung shadows.",
    "Lung Opacity":"Diffuse hazy lung regions.",
    "Nodule/Mass":"Rounded lung lesion.",
    "Other lesion":"Unclassified abnormality.",
    "Pleural effusion":"Fluid around lungs.",
    "Pleural thickening":"Thickened pleural lining.",
    "Pneumothorax":"Collapsed lung due to air.",
    "Pulmonary fibrosis":"Lung scarring.",
    "No finding":"No abnormality detected."
}

# =========================
# LOAD MODELS
# =========================
CLASSIFICATION_MODEL_PATH = "models/clinicscan_classifier_15class.pth"
@torch.no_grad()
def load_classifier():
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(2048, 15)
    model.load_state_dict(torch.load(CLASSIFICATION_MODEL_PATH, map_location="cpu"))
    model.eval()
    return model

classifier = load_classifier()
yolo_model = YOLO("models/best.pt")

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
        image_file = request.files["image"]
        image = Image.open(image_file).convert("RGB")

        # -------- Classification --------
        input_tensor = transform(image).unsqueeze(0)
        logits = classifier(input_tensor)
        pred_idx = torch.argmax(logits, dim=1).item()
        class_result = CLASS_NAMES[pred_idx]

        # -------- YOLO Detection --------
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        results = yolo_model(
            img_cv,
            conf=0.5,
            iou=0.4,
            max_det=5
        )[0]

        best_boxes = {}
        detections = []

        if results.boxes is not None:
            for box in results.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                cls_name = yolo_model.names[cls_id]

                if cls_name not in best_boxes or conf > best_boxes[cls_name]["conf"]:
                    best_boxes[cls_name] = {
                        "box": box.xyxy[0].cpu().numpy().astype(int),
                        "conf": round(conf, 2)
                    }

        # -------- Draw boxes --------
        for disease, data in best_boxes.items():
            x1, y1, x2, y2 = data["box"]
            conf = data["conf"]

            cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                img_cv,
                f"{disease} {conf}",
                (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

            detections.append((disease, conf))

        cv2.imwrite(OUTPUT_IMAGE_PATH, img_cv)

        definitions = {
            d: DISEASE_DEFINITIONS.get(d, "Definition not available.")
            for d, _ in detections
        }

        return render_template(
            "index.html",
            classification=class_result,
            detections=detections,
            definitions=definitions,
            image_path=OUTPUT_IMAGE_PATH
        )

    return render_template("index.html")

# =========================
# MAIN
# =========================
app.run(debug=True)

