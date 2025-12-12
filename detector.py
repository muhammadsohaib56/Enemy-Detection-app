import os
import cv2
import numpy as np
from ultralytics import YOLO

# ==================== AUTO DOWNLOAD YOLOv8x ====================
MODEL_NAME = "yolov8x.pt"
MODEL_PATH = os.path.join("models", MODEL_NAME)

os.makedirs("models", exist_ok=True)

if not os.path.isfile(MODEL_PATH):
    print(f"Downloading {MODEL_NAME}... (first run only)")
    YOLO(MODEL_NAME)
    print("Model downloaded successfully!")

yolo = YOLO(MODEL_PATH)
print("YOLOv8x model loaded successfully!")


def is_shadow_or_dark_artifact(image, x1, y1, x2, y2):
    """
    Detect if the detection is actually a shadow or dark artifact
    Shadows have:
    - Very low brightness
    - Low contrast
    - No distinct color information
    - Uniform darkness
    """
    region = image[y1:y2, x1:x2]
    
    if region.size == 0:
        return False
    
    # Convert to grayscale and HSV
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    
    # Check average brightness (V channel in HSV)
    avg_brightness = np.mean(hsv[:, :, 2])
    
    # Check if region is predominantly dark
    dark_pixels = np.sum(gray < 50) / gray.size
    
    # Check color saturation (shadows have very low saturation)
    avg_saturation = np.mean(hsv[:, :, 1])
    
    # Calculate contrast using standard deviation
    contrast = np.std(gray)
    
    # Shadow detection criteria
    is_very_dark = avg_brightness < 60
    is_mostly_dark = dark_pixels > 0.6
    is_low_saturation = avg_saturation < 30
    is_low_contrast = contrast < 25
    
    # If all conditions met, likely a shadow
    if is_very_dark and is_mostly_dark and is_low_saturation and is_low_contrast:
        return True
    
    return False


def detect_pink_enemy_bar(image, x1, y1, x2, y2):
    """
    ENHANCED: Specifically detect FF00FF (magenta/pink) horizontal bar above enemy head
    This function is laser-focused on finding the exact enemy color bar
    """
    h, w = image.shape[:2]
    
    # Define multiple search regions above head
    # Check wider area and multiple heights for better detection
    search_regions = [
        (max(y1 - 18, 0), max(y1 - 3, 0)),   # Very close to head
        (max(y1 - 28, 0), max(y1 - 13, 0)),  # Medium distance
        (max(y1 - 38, 0), max(y1 - 23, 0)),  # Far from head
        (max(y1 - 48, 0), max(y1 - 33, 0))   # Very far (for distant enemies)
    ]
    
    box_width = x2 - x1
    
    # Expand horizontal search to catch partial bars
    x_left = max(x1 - int(box_width * 0.15), 0)
    x_right = min(x2 + int(box_width * 0.15), w)
    
    best_pink_confidence = 0.0
    
    for top, bottom in search_regions:
        if top >= bottom or x_left >= x_right:
            continue
            
        bar_region = image[top:bottom, x_left:x_right]
        
        if bar_region.size == 0:
            continue
        
        # Convert to HSV and BGR for multi-approach detection
        hsv = cv2.cvtColor(bar_region, cv2.COLOR_BGR2HSV)
        bgr = bar_region.copy()
        
        # ===== METHOD 1: HSV Pink/Magenta Detection =====
        # FF00FF in HSV is around H=300° (which is 150 in OpenCV's 0-180 scale)
        # Primary magenta range
        pink_mask1 = cv2.inRange(hsv, np.array([140, 100, 150]), np.array([170, 255, 255]))
        
        # Extended magenta range (wider hue tolerance)
        pink_mask2 = cv2.inRange(hsv, np.array([135, 80, 130]), np.array([175, 255, 255]))
        
        # Purple-pink range (for slight color variations)
        pink_mask3 = cv2.inRange(hsv, np.array([130, 70, 120]), np.array([160, 255, 255]))
        
        # Combine all HSV masks
        hsv_pink_mask = cv2.bitwise_or(pink_mask1, pink_mask2)
        hsv_pink_mask = cv2.bitwise_or(hsv_pink_mask, pink_mask3)
        
        # ===== METHOD 2: Direct BGR Magenta Detection =====
        # FF00FF = BGR(255, 0, 255) - high blue, low green, high red
        # Look for high B and R, low G
        b, g, r = cv2.split(bgr)
        
        # Create mask where B > 120, R > 120, G < 80
        magenta_bgr_mask = np.logical_and.reduce([
            b > 120,
            r > 120,
            g < 80
        ]).astype(np.uint8) * 255
        
        # ===== METHOD 3: Color Distance to FF00FF =====
        # Calculate Euclidean distance to pure magenta (255, 0, 255)
        target_magenta = np.array([255, 0, 255], dtype=np.float32)
        bgr_float = bgr.astype(np.float32)
        
        color_distance = np.sqrt(np.sum((bgr_float - target_magenta) ** 2, axis=2))
        # Pixels within distance 100 are considered magenta-like
        distance_mask = (color_distance < 100).astype(np.uint8) * 255
        
        # ===== COMBINE ALL METHODS =====
        combined_mask = cv2.bitwise_or(hsv_pink_mask, magenta_bgr_mask)
        combined_mask = cv2.bitwise_or(combined_mask, distance_mask)
        
        # Calculate pink ratio
        pink_ratio = np.count_nonzero(combined_mask) / combined_mask.size
        
        # ===== CHECK FOR HORIZONTAL BAR PATTERN =====
        # Enemy name bars are horizontal, so check for horizontal continuity
        if pink_ratio > 0.05:  # Only check pattern if we have some pink
            # Check if pink pixels form horizontal lines
            horizontal_projection = np.sum(combined_mask, axis=1)
            max_horizontal = np.max(horizontal_projection) if len(horizontal_projection) > 0 else 0
            
            # If there's a strong horizontal line of pink, boost confidence
            if max_horizontal > combined_mask.shape[1] * 180:  # At least 70% width
                pink_ratio *= 1.5  # Boost confidence for horizontal bars
        
        # Update best confidence
        if pink_ratio > best_pink_confidence:
            best_pink_confidence = pink_ratio
    
    return best_pink_confidence


def detect_blue_teammate_bar(image, x1, y1, x2, y2):
    """
    Specifically detect blue horizontal bar above teammate head
    """
    h, w = image.shape[:2]
    
    search_regions = [
        (max(y1 - 18, 0), max(y1 - 3, 0)),
        (max(y1 - 28, 0), max(y1 - 13, 0)),
        (max(y1 - 38, 0), max(y1 - 23, 0)),
        (max(y1 - 48, 0), max(y1 - 33, 0))
    ]
    
    box_width = x2 - x1
    x_left = max(x1 - int(box_width * 0.15), 0)
    x_right = min(x2 + int(box_width * 0.15), w)
    
    best_blue_confidence = 0.0
    
    for top, bottom in search_regions:
        if top >= bottom or x_left >= x_right:
            continue
            
        bar_region = image[top:bottom, x_left:x_right]
        
        if bar_region.size == 0:
            continue
        
        hsv = cv2.cvtColor(bar_region, cv2.COLOR_BGR2HSV)
        bgr = bar_region.copy()
        
        # Multiple blue detection ranges
        blue_mask1 = cv2.inRange(hsv, np.array([95, 100, 130]), np.array([130, 255, 255]))
        blue_mask2 = cv2.inRange(hsv, np.array([90, 70, 100]), np.array([135, 255, 255]))
        
        # Direct BGR blue check
        b, g, r = cv2.split(bgr)
        blue_bgr_mask = np.logical_and.reduce([
            b > 120,
            g < 150,
            r < 100
        ]).astype(np.uint8) * 255
        
        # Combine masks
        combined_mask = cv2.bitwise_or(blue_mask1, blue_mask2)
        combined_mask = cv2.bitwise_or(combined_mask, blue_bgr_mask)
        
        blue_ratio = np.count_nonzero(combined_mask) / combined_mask.size
        
        # Check for horizontal pattern
        if blue_ratio > 0.05:
            horizontal_projection = np.sum(combined_mask, axis=1)
            max_horizontal = np.max(horizontal_projection) if len(horizontal_projection) > 0 else 0
            
            if max_horizontal > combined_mask.shape[1] * 180:
                blue_ratio *= 1.5
        
        if blue_ratio > best_blue_confidence:
            best_blue_confidence = blue_ratio
    
    return best_blue_confidence


def is_mannequin_or_static(image, x1, y1, x2, y2):
    """
    Detect if entity is likely a mannequin/clone
    """
    entity_region = image[y1:y2, x1:x2]
    
    if entity_region.size == 0:
        return False
    
    # Calculate color variance
    gray = cv2.cvtColor(entity_region, cv2.COLOR_BGR2GRAY)
    variance = np.var(gray)
    
    # Low variance suggests mannequin
    if variance < 180:
        return True
    
    # Check for predominantly gray/neutral colors
    hsv = cv2.cvtColor(entity_region, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    avg_saturation = np.mean(saturation)
    
    # Low saturation = likely mannequin
    if avg_saturation < 35:
        return True
    
    return False


def has_human_features(image, x1, y1, x2, y2):
    """
    Check if detection has human-like features (not just shadow)
    """
    region = image[y1:y2, x1:x2]
    
    if region.size == 0:
        return False
    
    # Convert to grayscale
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    
    # Use edge detection to find structure
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.count_nonzero(edges) / edges.size
    
    # Humans have more edge structure than shadows
    if edge_density < 0.02:  # Very few edges = likely shadow
        return False
    
    # Check for color variety (humans wear colored clothes)
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    
    # Count pixels with some color (not just gray/black)
    colored_pixels = np.sum(saturation > 20) / saturation.size
    
    if colored_pixels < 0.15:  # Less than 15% colored = likely shadow
        return False
    
    return True


def detect_enemies(image_path, output_dir="outputs", min_confidence=0.35):
    """
    Main detection function with enhanced enemy detection and shadow filtering
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Run YOLO detection
    results = yolo.predict(source=image, conf=min_confidence, verbose=False)[0]
    
    detections = []
    debug_info = {
        "total_detections": 0,
        "filtered_shadows": 0,
        "filtered_mannequins": 0,
        "filtered_teammates": 0,
        "enemies_found": 0,
        "bots_found": 0
    }

    for box in results.boxes:
        cls_id = int(box.cls[0])
        if cls_id != 0:  # Only process person class
            continue

        debug_info["total_detections"] += 1
        
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        width = x2 - x1
        height = y2 - y1
        
        # Size filter for distant enemies
        if width < 25 or height < 35:
            continue
        
        # CRITICAL: Filter out shadows first
        if is_shadow_or_dark_artifact(image, x1, y1, x2, y2):
            debug_info["filtered_shadows"] += 1
            continue
        
        # Check if it has human features (not shadow)
        if not has_human_features(image, x1, y1, x2, y2):
            debug_info["filtered_shadows"] += 1
            continue
        
        # Check if mannequin/clone
        if is_mannequin_or_static(image, x1, y1, x2, y2):
            debug_info["filtered_mannequins"] += 1
            continue
        
        # Detect color bars with enhanced detection
        pink_confidence = detect_pink_enemy_bar(image, x1, y1, x2, y2)
        blue_confidence = detect_blue_teammate_bar(image, x1, y1, x2, y2)
        
        # Classification logic
        # PRIORITY 1: If blue bar detected -> TEAMMATE (skip)
        if blue_confidence > 0.12:  # Lowered threshold for better teammate detection
            debug_info["filtered_teammates"] += 1
            continue
        
        # PRIORITY 2: If pink bar detected -> ENEMY (high priority)
        if pink_confidence > 0.08:  # Lowered threshold to catch more real enemies
            role = "Enemy"
            color = (0, 0, 255)  # Bright red
            confidence = pink_confidence
            debug_info["enemies_found"] += 1
        else:
            # No clear bar detected -> likely BOT
            role = "Enemy (Bot)"
            color = (0, 100, 255)  # Orange-red
            confidence = 0.5  # Default bot confidence
            debug_info["bots_found"] += 1
        
        detections.append({
            "bbox": (x1, y1, x2, y2),
            "role": role,
            "color": color,
            "confidence": confidence
        })

    # Draw all valid detections
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        
        # Thicker box for confirmed enemies
        thickness = 4 if det["role"] == "Enemy" else 3
        
        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), det["color"], thickness)
        
        # Prepare label
        label = f"{det['role']}"
        if det["confidence"] > 0 and det["role"] == "Enemy":
            label += f" ({det['confidence']:.0%})"
        
        # Draw label background
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(image, (x1, y1 - text_h - 10), (x1 + text_w + 6, y1), det["color"], -1)
        
        # Draw label text
        cv2.putText(image, label, (x1 + 3, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Save output
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.basename(image_path)
    name, ext = os.path.splitext(filename)
    output_path = os.path.join(output_dir, f"{name}_output{ext}")
    
    cv2.imwrite(output_path, image)
    
    # Convert for display
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Print debug information
    print(f"\n{'='*50}")
    print(f"Detection Summary:")
    print(f"{'='*50}")
    print(f"Total YOLO detections: {debug_info['total_detections']}")
    print(f"Filtered shadows: {debug_info['filtered_shadows']}")
    print(f"Filtered mannequins: {debug_info['filtered_mannequins']}")
    print(f"Filtered teammates: {debug_info['filtered_teammates']}")
    print(f"Enemies found (pink bar): {debug_info['enemies_found']}")
    print(f"Bots found (no bar): {debug_info['bots_found']}")
    print(f"Total threats detected: {len(detections)}")
    print(f"{'='*50}\n")
    
    return image_rgb, output_path