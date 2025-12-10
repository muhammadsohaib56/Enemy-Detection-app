import os
import cv2
import numpy as np
from ultralytics import YOLO

# ==================== AUTO DOWNLOAD YOLOv8x (WORKS PERFECTLY ON STREAMLIT) ====================
MODEL_NAME = "yolov8x.pt"
MODEL_PATH = os.path.join("models", MODEL_NAME)

# Create models folder
os.makedirs("models", exist_ok=True)

# Download model only if not already present (happens once)
if not os.path.isfile(MODEL_PATH):
    print(f"Downloading {MODEL_NAME} from Ultralytics hub... (first run only, ~130MB)")
    YOLO(MODEL_NAME)  # This automatically downloads and saves to models/yolov8x.pt
    print("Model downloaded successfully!")

# Load the model
yolo = YOLO(MODEL_PATH)
print("YOLOv8x model loaded successfully!")


def detect_bar_above_head(image, x1, y1, x2):
    """Detect pink (enemy) or blue (teammate) name bar above head"""
    h, w = image.shape[:2]

    top = max(y1 - 12, 0)
    bottom = max(y1 - 2, 0)

    if top >= bottom or x1 >= x2:
        return "bot"

    bar_region = image[top:bottom, x1:x2]
    if bar_region.size == 0:
        return "bot"

    hsv = cv2.cvtColor(bar_region, cv2.COLOR_BGR2HSV)

    # Pink range (FF00FF)
    pink_mask = cv2.inRange(hsv, np.array([140, 100, 120]), np.array([170, 255, 255]))
    # Blue teammate range
    blue_mask = cv2.inRange(hsv, np.array([95, 100, 120]), np.array([130, 255, 255]))

    pink_ratio = pink_mask.mean() / 255.0
    blue_ratio = blue_mask.mean() / 255.0

    if pink_ratio > 0.10 and blue_ratio < 0.02:
        return "enemy"
    if blue_ratio > 0.10 and pink_ratio < 0.02:
        return "teammate"
    return "bot"


def detect_enemies(image_path, output_dir="outputs"):
    """
    Main function: detects people → draws boxes → saves result → returns image for Streamlit
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Run inference
    results = yolo.predict(source=image, conf=0.45, verbose=False)[0]

    for box in results.boxes:
        cls_id = int(box.cls[0])
        if cls_id != 0:  # 0 = person in COCO dataset
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        width = x2 - x1
        height = y2 - y1

        # Filter too small detections
        if width < 40 or height < 50:
            continue

        role = detect_bar_above_head(image, x1, y1, x2)

        # Black box + label for teammate, Red for enemy/bot
        if role == "teammate":
            color = (0, 0, 0)      # Black
            label = "Teammate"
        else:
            color = (0, 0, 255)    # Red
            label = "Enemy"

        # Draw box and label
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
        cv2.putText(image, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Save output image
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.basename(image_path)
    name, ext = os.path.splitext(filename)
    output_path = os.path.join(output_dir, f"{name}_output{ext}")

    cv2.imwrite(output_path, image)

    # Convert BGR → RGB for Streamlit display
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image_rgb, output_path