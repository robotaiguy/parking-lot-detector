#!/usr/bin/env python3
"""
PARKING DETECTOR - CRITICAL FIX
Bug: Bounds check used wrong variables (w, h instead of image width/height)
"""

import cv2
import numpy as np
import os
from pathlib import Path

INPUT_PATH = "./input_images/"
OUTPUT_PATH = "/mnt/c/wsl_outputs/"

def process_parking_lot(img_path, output_path):
    """Process one parking lot image"""
    img_name = Path(img_path).stem
    print(f"\n{'='*70}\nPROCESSING: {img_name}\n{'='*70}")
    
    img_dir = os.path.join(output_path, f"image_{img_name}")
    os.makedirs(img_dir, exist_ok=True)
    
    def save(img, name):
        cv2.imwrite(os.path.join(img_dir, f"{name}.jpg"), img)
    
    image = cv2.imread(str(img_path))
    if image is None:
        return None
    
    img_h, img_w = image.shape[:2]
    save(image, "00_original")
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    save(gray, "01_grayscale")
    
    # Detect lines
    print("  Detecting parking lines...")
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    white_mask = cv2.inRange(hsv, np.array([0, 0, 190]), np.array([180, 40, 255]))
    yellow_mask = cv2.inRange(hsv, np.array([15, 80, 80]), np.array([35, 255, 255]))
    line_mask = cv2.bitwise_or(white_mask, yellow_mask)
    
    kernel = np.ones((3,3), np.uint8)
    line_mask = cv2.morphologyEx(line_mask, cv2.MORPH_CLOSE, kernel)
    line_mask = cv2.morphologyEx(line_mask, cv2.MORPH_OPEN, kernel)
    save(line_mask, "02_lines")
    
    # Hough lines
    print("  Hough transform...")
    lines = cv2.HoughLinesP(line_mask, 1, np.pi/180, 50, minLineLength=40, maxLineGap=10)
    
    if lines is None:
        print("  ERROR: No lines detected")
        return None
    
    viz = image.copy()
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(viz, (x1, y1), (x2, y2), (0, 255, 0), 1)
    save(viz, "03_all_lines")
    print(f"  Detected {len(lines)} line segments")
    
    # Filter vertical lines
    print("  Filtering vertical lines...")
    vertical_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        
        if dy > dx * 1.5:  # Vertical
            length = np.sqrt(dx*dx + dy*dy)
            if length > 30:
                vertical_lines.append({
                    'x': (x1 + x2) / 2,
                    'y_min': min(y1, y2),
                    'y_max': max(y1, y2)
                })
    
    if len(vertical_lines) < 2:
        print("  ERROR: Not enough vertical lines")
        return None
    
    viz = image.copy()
    for vl in vertical_lines:
        x = int(vl['x'])
        cv2.line(viz, (x, int(vl['y_min'])), (x, int(vl['y_max'])), (255, 255, 0), 2)
    save(viz, "04_vertical_lines")
    print(f"  Found {len(vertical_lines)} vertical lines")
    
    # Remove duplicates
    vertical_lines.sort(key=lambda l: l['x'])
    unique_lines = [vertical_lines[0]]
    for vl in vertical_lines[1:]:
        if vl['x'] - unique_lines[-1]['x'] > 15:
            unique_lines.append(vl)
    
    print(f"  Unique lines: {len(unique_lines)}")
    
    # Create spaces
    print("  Creating parking spaces...")
    spaces = []
    for i in range(len(unique_lines) - 1):
        l1 = unique_lines[i]
        l2 = unique_lines[i+1]
        
        x1 = int(l1['x'])
        x2 = int(l2['x'])
        width = x2 - x1
        
        if 40 < width < 150:
            y_min = max(l1['y_min'], l2['y_min'])
            y_max = min(l1['y_max'], l2['y_max'])
            height = abs(y_max - y_min)
            spaces.append((x1, int(y_min), width, int(height)))
    
    viz = image.copy()
    for x, y, w, h in spaces:
        cv2.rectangle(viz, (x, y), (x+w, y+h), (255, 0, 0), 2)
    save(viz, "05_detected_spaces")
    print(f"  Generated {len(spaces)} parking spaces")
    
    if len(spaces) == 0:
        print("  ERROR: No spaces generated")
        return None
    
    # Occupancy detection
    print("  Detecting occupancy...")
    results = []
    
    for x, y, w, h in spaces:
        # FIX: Use img_w and img_h, not w and h
        if x < 0 or y < 0 or x+w > img_w or y+h > img_h or w <= 0 or h <= 0:
            print(f"    Skipping invalid space: ({x},{y},{w},{h})")
            continue
        
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = image[y:y+h, x:x+w]
        
        if roi_gray.size == 0:
            continue
        
        # Features
        mean_int = np.mean(roi_gray)
        std_dev = np.std(roi_gray)
        
        edges = cv2.Canny(roi_gray, 40, 120)
        edge_density = np.sum(edges > 0) / edges.size
        
        hsv_roi = cv2.cvtColor(roi_color, cv2.COLOR_BGR2HSV)
        white_in_roi = cv2.inRange(hsv_roi, np.array([0, 0, 190]), np.array([180, 40, 255]))
        line_visible = np.sum(white_in_roi > 0) / white_in_roi.size
        
        # Score
        score = 0
        if mean_int < 120: score += 3
        if std_dev > 25: score += 2
        if edge_density > 0.15: score += 2
        if line_visible < 0.05: score += 2
        
        is_occupied = score >= 5
        results.append({
            'bbox': (x, y, w, h),
            'occupied': is_occupied
        })
    
    # Visualize
    viz = image.copy()
    for r in results:
        x, y, w, h = r['bbox']
        color = (0, 0, 255) if r['occupied'] else (0, 255, 0)
        cv2.rectangle(viz, (x, y), (x+w, y+h), color, 3)
    save(viz, "06_occupancy")
    
    # Final
    final = image.copy()
    total = len(results)
    occupied = sum(1 for r in results if r['occupied'])
    empty = total - occupied
    
    for r in results:
        x, y, w, h = r['bbox']
        color = (0, 0, 255) if r['occupied'] else (0, 255, 0)
        cv2.rectangle(final, (x, y), (x+w, y+h), color, 3)
    
    cv2.rectangle(final, (10, 10), (500, 160), (0, 0, 0), -1)
    cv2.rectangle(final, (10, 10), (500, 160), (255, 255, 255), 3)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(final, f"Total Parking Spaces: {total}",
               (25, 50), font, 0.9, (255, 255, 255), 2)
    cv2.putText(final, f"Empty Spaces: {empty}",
               (25, 90), font, 0.9, (0, 255, 0), 2)
    cv2.putText(final, f"Occupied by Vehicles: {occupied}",
               (25, 130), font, 0.9, (0, 0, 255), 2)
    
    save(final, "07_FINAL_RESULT")
    
    print(f"\n{'='*70}")
    print(f"RESULTS: {img_name}")
    print(f"{'='*70}")
    print(f"Total Parking Spaces:    {total}")
    print(f"Empty Spaces:            {empty}")
    print(f"Occupied by Vehicles:    {occupied}")
    print(f"{'='*70}\n")
    
    return {'filename': img_name, 'total': total, 'empty': empty, 'occupied': occupied}


# MAIN
if __name__ == "__main__":
    print("\n" + "="*70)
    print("PARKING LOT DETECTOR - CRITICAL BUG FIX")
    print("="*70)
    
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    image_files = (list(Path(INPUT_PATH).glob("*.jpg")) +
                  list(Path(INPUT_PATH).glob("*.png")))
    
    if not image_files:
        print("\nERROR: No images found")
        exit(1)
    
    print(f"\nProcessing {len(image_files)} images\n")
    
    all_results = []
    for img_path in sorted(image_files):
        result = process_parking_lot(img_path, OUTPUT_PATH)
        if result:
            all_results.append(result)
    
    print("\n" + "="*70)
    print("PROCESSING COMPLETE - SUMMARY")
    print("="*70)
    for r in all_results:
        print(f"{r['filename']:20s} | Total: {r['total']:3d} | "
              f"Empty: {r['empty']:3d} | Occupied: {r['occupied']:3d}")
    print("="*70 + "\n")
