# =========================
# CPU ONLY (IMPORTANT)
# =========================
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# =========================
# IMPORTS
# =========================
import gdown
import numpy as np
import cv2
from PIL import Image
from flask import Flask, render_template, request
from ultralytics import YOLO

# =========================
# PATHS
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
STATIC_DIR = os.path.join(BASE_DIR, "static")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

YOLO_PATH = os.path.join(MODEL_DIR, "best.pt")
OUTPUT_IMAGE_PATH = os.path.join(STATIC_DIR, "output.png")

# =========================
# GOOGLE DRIVE MODEL
# =========================
YOLO_ID = "1xzYVtQKGBvle7PPi4-XgwErhe7kSOiAm"

def download_model():
    if not os.path.exists(YOLO_PATH):
        gdown.download(
            f"https://drive.google.com/file/d/1xzYVtQKGBvle7PPi4-XgwErhe7kSOiAm/view?usp=drive_link",
            YOLO_PATH,
            quiet=False
        )

download_model()

# =========================
# FLASK APP
# =========================
app = Flask(__name__)

# =========================
# DISEASE DEFINITIONS
# =========================
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
# LOAD YOLO (LAZY)
# =========================
yolo_model = None

def get_yolo():
    global yolo_model
    if yolo_model is None:
        yolo_model = YOLO(YOLO_PATH)
    return yolo_model

# =========================
# ROUTES
# =========================
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        yolo = get_yolo()

        image = Image.open(request.files["image"]).convert("RGB")
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        results = yolo(
            img_cv,
            conf=0.5,
            iou=0.4,
            max_det=5,
            device="cpu"
        )[0]

        best_boxes = {}
        detections = []

        if results.boxes is not None:
            for box in results.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                name = yolo.names[cls_id]

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
                img_cv,
                f"{disease} {conf}",
                (x1, y1 - 10),
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
            detections=detections,
            definitions=definitions,
            image_path="static/output.png"
        )

    return render_template("index.html")

# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
