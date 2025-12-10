import os
import cv2
import numpy as np
from ultralytics import YOLO

# ==================== AUTO-DOWNLOAD YOLOv8x MODEL ====================
# This fixes the "model not found" error on Streamlit Cloud
# The first time the app runs, it will automatically download yolov8x.pt (~130MB)
# After that, it loads instantly from cache

MODEL_NAME = "yolov8x.pt"

# Create models folder if it doesn't exist
os.makedirs("models", exist_ok=True)
model_path = os.path.join("models", MODEL_NAME)

# Download only if not already present
if not os.path.exists(model_path):
    print(f"Downloading {MODEL_NAME} (this takes ~30-60 seconds on first run)...")
    YOLO(MODEL_NAME)  # This triggers official Ultralytics download and saves it automatically
    print("Model downloaded and cached!")

# Now load the model
yolo = YOLO(model_path)
print("YOLOv8x model loaded successfully!")


def detect_bar_above_head(image, x1, y1, x2):
    """
    Detects pink (enemy) or blue (teammate) bar directly above the head.
    Region: 2–12 pixels above the detected person's head.
    """
    h, w = image.shape[:2]

    bar_top = max(y1 - 12, 0)
    bar_bottom = max(y1 - 2, 0)

    # Safety check
    if bar_top >= bar_bottom or x1 >= x2:
        return "bot"

    bar_region = image[bar_top:bar_bottom, x1:x2]

    if bar_region.size == 0:
        return "bot"

    hsv = cv2.cvtColor(bar_region, cv2.COLOR_BGR2HSV)

    # Pink FF00FF range
    lower_pink = np.array([140, 100, 120], dtype=np.uint8)
    upper_pink = np.array([170, 255, 255], dtype=np.uint8)

    # Blue teammate range
    lower_blue = np.array([95, 100, 120], dtype=np.uint8)
    upper_blue = np.array([130, 255, 255], dtype=np.uint8)

    pink_mask = cv2.inRange(hsv, lower_pink, upper_pink)
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    pink_ratio = pink_mask.mean() / 255.0
    blue_ratio = blue_mask.mean() / 255.0

    if pink_ratio > 0.10 and blue_ratio < 0.02:
        return "enemy"
    if blue_ratio > 0.10 and pink_ratio < 0.02:
        return "teammate"

    return "bot"


def detect_enemies(image_path, output_dir="outputs"):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    # Run YOLO detection
    results = yolo.predict(image, conf=0.45, verbose=False)[0]
    h_img, w_img = image.shape[:2]

    for det in results.boxes:
        if int(det.cls) != 0:  # class 0 = person
            continue

        x1, y1, x2, y2 = map(int, det.xyxy[0].cpu().numpy())

        # Clamp coordinates
        x1 = max(0, min(x1, w_img - 1))
        x2 = max(0, min(x2, w_img - 1))
        y1 = max(0, min(y1, h_img - 1))
        y2 = max(0, min(y2, h_img - 1))

        # Filter small detections
        if (x2 - x1) < 40 or (y2 - y1) < 50:
            continue

        role = detect_bar_above_head(image, x1, y1, x2)

        if role == "teammate":
            color = (0, 0, 0)      # Black box
            label = "Teammate"
        else:
            color = (0, 0, 255)    # Red box
            label = "Enemy"

        # Draw bounding box and label
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
        cv2.putText(image, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Save output
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.basename(image_path)
    name, ext = os.path.splitext(base)
    out_path = os.path.join(output_dir, f"{name}_output{ext}")
    cv2.imwrite(out_path, image)

    return image, out_path