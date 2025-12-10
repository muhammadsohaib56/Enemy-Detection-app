import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Load YOLO model
yolo = YOLO("models/yolov8x.pt")  


def detect_bar_above_head(image, x1, y1, x2):
    """
    NEW FIXED BAR DETECTION:
    Look for the horizontal bar OUTSIDE the person crop,
    directly ABOVE the head.

    Region: 10–12 px height above y1.
    """

    h, w = image.shape[:2]

    bar_top = max(y1 - 12, 0)
    bar_bottom = max(y1 - 2, 0)

    bar_region = image[bar_top:bar_bottom, x1:x2]

    if bar_region.size == 0:
        return "bot"

    hsv = cv2.cvtColor(bar_region, cv2.COLOR_BGR2HSV)

    # Pink FF00FF
    lower_pink = np.array([140, 100, 120], dtype=np.uint8)
    upper_pink = np.array([170, 255, 255], dtype=np.uint8)

    # Blue teammate
    lower_blue = np.array([95, 100, 120], dtype=np.uint8)
    upper_blue = np.array([130, 255, 255], dtype=np.uint8)

    pink_mask = cv2.inRange(hsv, lower_pink, upper_pink)
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    pink_ratio = pink_mask.mean() / 255.0
    blue_ratio = blue_mask.mean() / 255.0

    # Strict thresholds
    if pink_ratio > 0.10 and blue_ratio < 0.02:
        return "enemy"
    if blue_ratio > 0.10 and pink_ratio < 0.02:
        return "teammate"

    # No bar
    return "bot"


def detect_enemies(image_path, output_dir="outputs"):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    results = yolo.predict(image, conf=0.45, verbose=False)[0]
    h_img, w_img = image.shape[:2]

    for det in results.boxes:
        if int(det.cls) != 0:  
            continue

        x1, y1, x2, y2 = det.xyxy[0].cpu().numpy().astype(int)

        # Clamp
        x1 = max(0, min(x1, w_img - 1))
        x2 = max(0, min(x2, w_img - 1))
        y1 = max(0, min(y1, h_img - 1))
        y2 = max(0, min(y2, h_img - 1))

        if (x2 - x1) < 40 or (y2 - y1) < 50:
            continue

        # DETECT BAR OUTSIDE PERSON
        role = detect_bar_above_head(image, x1, y1, x2)

        if role == "teammate":
            color = (0, 0, 0)
            label = "Teammate"
        else:
            color = (0, 0, 255)
            label = "Enemy"

        # Draw
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
        cv2.putText(
            image, label, (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
        )

    os.makedirs(output_dir, exist_ok=True)
    base = os.path.basename(image_path)
    name, ext = os.path.splitext(base)
    out_path = os.path.join(output_dir, f"{name}_output{ext}")
    cv2.imwrite(out_path, image)

    return image, out_path
