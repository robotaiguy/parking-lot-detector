import cv2
import numpy as np
from typing import List, Tuple, Dict
import os

class HybridParkingDetector:
    """
    Hybrid parking detection system combining:
    - Perimeter/Grid reconstruction approach (lines-first)
    - Object segmentation approach (objects-first)
    """
    
    def __init__(self, debug=True):
        self.debug = debug
        self.output_dir = '/mnt/c/wsl_outputs/'
        os.makedirs(self.output_dir, exist_ok=True)
        
        # =================================================================
        # TUNING PARAMETERS - Adjust these for your images
        # =================================================================
        
        # --- METHOD SELECTION ---
        self.use_perimeter_method = True      # Try grid reconstruction first
        self.use_intersection_method = True   # Fallback to intersection method
        self.min_spots_threshold = 5          # If perimeter finds < this, use intersection
        
        # --- LINE DETECTION (Perimeter Method) ---
        self.tophat_size = 25                 # Size for top-hat transform (15-35)
        self.adaptive_block_size = 35         # Block size for adaptive threshold (21-51, odd)
        self.adaptive_c = -5                  # Constant subtracted (-10 to 5)
        
        # --- LINE FILTERING (Stick Sieve) ---
        self.min_line_aspect = 3.0            # Lines must be this elongated (2.5-4.0)
        self.min_line_solidity = 0.85         # Lines must be this solid (0.75-0.95)
        self.min_line_area = 50               # Minimum line area in pixels
        
        # --- GRID RECONSTRUCTION ---
        self.dilate_iterations = 3            # Connect broken lines (2-5)
        self.close_iterations = 3             # Seal parking spot "rooms" (2-5)
        
        # --- SPOT FILTERING ---
        self.min_spot_area = 800              # Minimum spot area (500-1500)
        self.max_spot_area = 25000            # Maximum spot area (15000-35000)
        self.min_spot_aspect = 1.0            # Min aspect ratio (0.8-1.5)
        self.max_spot_aspect = 5.0            # Max aspect ratio (4.0-6.0)
        self.min_spot_solidity = 0.75         # Rectangularity (0.70-0.85)
        
        # --- INTERSECTION METHOD (Fallback) ---
        self.hough_threshold = 40             # Hough line threshold (30-60)
        self.hough_min_length = 25            # Minimum line length (20-40)
        self.hough_max_gap = 15               # Max gap in line (10-20)
        self.line_merge_threshold = 30        # Distance to merge parallel lines (20-40)
        
        # --- VEHICLE DETECTION (Occupancy) ---
        self.vehicle_sat_threshold = 40       # Saturation threshold (30-60)
        self.vehicle_dark_threshold = 60      # Darkness threshold (50-80)
        self.occupancy_threshold = 0.15       # % overlap to be occupied (0.10-0.25)
        
        # State
        self.vehicle_mask = None
        self.method_used = None
        
    def log(self, message):
        """Print log message."""
        print(f"  {message}")
    
    def save_debug(self, img, base_name, label):
        """Save debug image if debug mode is on."""
        if self.debug:
            filename = f"{os.path.splitext(os.path.basename(base_name))[0]}_{label}.jpg"
            path = os.path.join(self.output_dir, filename)
            cv2.imwrite(path, img)
            self.log(f"Saved: {label}")
    
    # ============================================================
    # VEHICLE SEGMENTATION (Used by both methods)
    # ============================================================
    
    def get_vehicle_mask(self, img: np.ndarray) -> np.ndarray:
        """Generate mask of likely vehicles for occupancy detection."""
        self.log("Generating vehicle mask...")
        
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 1. Saturation (colorful cars)
        _, sat_mask = cv2.threshold(hsv[:,:,1], self.vehicle_sat_threshold, 255, cv2.THRESH_BINARY)
        
        # 2. Dark regions (dark cars/shadows)
        _, dark_mask = cv2.threshold(gray, self.vehicle_dark_threshold, 255, cv2.THRESH_BINARY_INV)
        
        # 3. Edges (windshields, grilles, details)
        edges = cv2.Canny(gray, 50, 150)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        edge_blob = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # Combine all cues
        mask = cv2.bitwise_or(sat_mask, dark_mask)
        mask = cv2.bitwise_or(mask, edge_blob)
        
        # Clean noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        return mask
    
    # ============================================================
    # METHOD 1: PERIMETER/GRID RECONSTRUCTION
    # ============================================================
    
    def detect_spots_perimeter(self, img: np.ndarray, img_name: str) -> List[Tuple]:
        """Detect parking spots using line isolation and grid reconstruction."""
        self.log("Method 1: Perimeter/Grid Reconstruction")
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Step 1: Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        self.save_debug(enhanced, img_name, "perimeter_01_enhanced")
        
        # Step 2: Top-hat transform (isolate bright, narrow features)
        kernel_tophat = cv2.getStructuringElement(cv2.MORPH_RECT, 
                                                   (self.tophat_size, self.tophat_size))
        tophat = cv2.morphologyEx(enhanced, cv2.MORPH_TOPHAT, kernel_tophat)
        self.save_debug(tophat, img_name, "perimeter_02_tophat")
        
        # Step 3: Adaptive threshold
        thresh = cv2.adaptiveThreshold(tophat, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, self.adaptive_block_size, 
                                       self.adaptive_c)
        self.save_debug(thresh, img_name, "perimeter_03_threshold")
        
        # Step 4: "Stick Sieve" - Keep only long, thin, solid lines
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        clean_lines = np.zeros_like(thresh)
        
        line_count = 0
        for c in contours:
            area = cv2.contourArea(c)
            if area < self.min_line_area:
                continue
            
            # Calculate solidity
            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)
            if hull_area == 0:
                continue
            solidity = area / hull_area
            
            # Calculate aspect ratio using rotated rectangle
            rect = cv2.minAreaRect(c)
            w, h = rect[1]
            if w == 0 or h == 0:
                continue
            short, long_side = min(w, h), max(w, h)
            if short == 0:
                continue
            aspect = long_side / short
            
            # Keep only valid parking lines (long, thin, solid)
            if aspect > self.min_line_aspect and solidity > self.min_line_solidity:
                cv2.drawContours(clean_lines, [c], -1, 255, -1)
                line_count += 1
        
        self.log(f"Found {line_count} valid parking lines")
        self.save_debug(clean_lines, img_name, "perimeter_04_clean_lines")
        
        # Step 5: Grid reconstruction - connect lines
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(clean_lines, dilate_kernel, iterations=self.dilate_iterations)
        self.save_debug(dilated, img_name, "perimeter_05_dilated")
        
        # Step 6: Close to seal parking spot "rooms"
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, close_kernel, 
                                  iterations=self.close_iterations)
        self.save_debug(closed, img_name, "perimeter_06_closed")
        
        # Step 7: Invert - lines become walls, spots become regions
        inverted = cv2.bitwise_not(closed)
        self.save_debug(inverted, img_name, "perimeter_07_inverted")
        
        # Step 8: Find parking spot regions
        contours, _ = cv2.findContours(inverted, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        spots = []
        for c in contours:
            area = cv2.contourArea(c)
            
            # Filter by area
            if area < self.min_spot_area or area > self.max_spot_area:
                continue
            
            # Calculate solidity
            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)
            if hull_area == 0:
                continue
            solidity = area / hull_area
            
            if solidity < self.min_spot_solidity:
                continue
            
            # Calculate aspect ratio
            rect = cv2.minAreaRect(c)
            (cx, cy), (w, h), angle = rect
            if w == 0 or h == 0:
                continue
            short = min(w, h)
            long_side = max(w, h)
            if short == 0:
                continue
            aspect = long_side / short
            
            if not (self.min_spot_aspect < aspect < self.max_spot_aspect):
                continue
            
            # Valid spot found - store as bounding box
            box = cv2.boxPoints(rect)
            box = np.int32(box)
            x_min = int(np.min(box[:, 0]))
            y_min = int(np.min(box[:, 1]))
            x_max = int(np.max(box[:, 0]))
            y_max = int(np.max(box[:, 1]))
            
            spots.append((x_min, y_min, x_max, y_max, rect))  # Include rect for drawing
        
        self.log(f"Detected {len(spots)} parking spots")
        return spots
    
    # ============================================================
    # METHOD 2: INTERSECTION-BASED (Fallback)
    # ============================================================
    
    def detect_lines_hough(self, img: np.ndarray, img_name: str) -> Dict:
        """Detect lines using Hough Transform."""
        self.log("Detecting lines with Hough Transform...")
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Threshold for white lines
        _, white_mask = cv2.threshold(enhanced, 160, 255, cv2.THRESH_BINARY)
        kernel = np.ones((2,2), np.uint8)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
        
        self.save_debug(white_mask, img_name, "intersection_01_white_mask")
        
        # Edge detection
        edges = cv2.Canny(white_mask, 50, 150, apertureSize=3)
        self.save_debug(edges, img_name, "intersection_02_edges")
        
        # Hough line detection
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 
                                threshold=self.hough_threshold,
                                minLineLength=self.hough_min_length, 
                                maxLineGap=self.hough_max_gap)
        
        if lines is None:
            return {'horizontal': [], 'vertical': []}
        
        # Visualize all lines
        lines_vis = img.copy()
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(lines_vis, (x1, y1), (x2, y2), (0, 255, 0), 1)
        self.save_debug(lines_vis, img_name, "intersection_03_all_lines")
        
        # Cluster into horizontal and vertical
        horizontal = []
        vertical = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            if length < 20:
                continue
            
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            
            if angle < 20 or angle > 160:
                horizontal.append(line[0])
            elif 70 < angle < 110:
                vertical.append(line[0])
        
        self.log(f"Found {len(horizontal)} horizontal, {len(vertical)} vertical lines")
        
        # Visualize clustered lines
        clustered_vis = img.copy()
        for line in horizontal:
            x1, y1, x2, y2 = line
            cv2.line(clustered_vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
        for line in vertical:
            x1, y1, x2, y2 = line
            cv2.line(clustered_vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
        self.save_debug(clustered_vis, img_name, "intersection_04_clustered")
        
        return {'horizontal': horizontal, 'vertical': vertical}
    
    def merge_parallel_lines(self, lines: List, is_horizontal: bool) -> List:
        """Merge lines that are close and parallel."""
        if not lines:
            return []
        
        # Sort by position
        if is_horizontal:
            lines = sorted(lines, key=lambda l: (l[1] + l[3]) / 2)
        else:
            lines = sorted(lines, key=lambda l: (l[0] + l[2]) / 2)
        
        merged = []
        current_group = [lines[0]]
        
        for i in range(1, len(lines)):
            if is_horizontal:
                prev_pos = (current_group[-1][1] + current_group[-1][3]) / 2
                curr_pos = (lines[i][1] + lines[i][3]) / 2
            else:
                prev_pos = (current_group[-1][0] + current_group[-1][2]) / 2
                curr_pos = (lines[i][0] + lines[i][2]) / 2
            
            if abs(curr_pos - prev_pos) < self.line_merge_threshold:
                current_group.append(lines[i])
            else:
                # Average the group
                avg_line = np.mean(current_group, axis=0).astype(int)
                merged.append(avg_line.tolist())
                current_group = [lines[i]]
        
        if current_group:
            avg_line = np.mean(current_group, axis=0).astype(int)
            merged.append(avg_line.tolist())
        
        return merged
    
    def detect_spots_intersection(self, img: np.ndarray, img_name: str) -> List[Tuple]:
        """Detect parking spots using line intersections."""
        self.log("Method 2: Intersection-based detection")
        
        # Detect lines
        lines_dict = self.detect_lines_hough(img, img_name)
        
        h_lines = lines_dict['horizontal']
        v_lines = lines_dict['vertical']
        
        if len(h_lines) < 2 or len(v_lines) < 2:
            self.log("Not enough lines for grid detection")
            return []
        
        # Merge parallel lines
        h_lines = self.merge_parallel_lines(h_lines, is_horizontal=True)
        v_lines = self.merge_parallel_lines(v_lines, is_horizontal=False)
        
        self.log(f"After merging: {len(h_lines)} horizontal, {len(v_lines)} vertical")
        
        # Visualize merged lines
        merged_vis = img.copy()
        for line in h_lines:
            x1, y1, x2, y2 = line
            cv2.line(merged_vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
        for line in v_lines:
            x1, y1, x2, y2 = line
            cv2.line(merged_vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
        self.save_debug(merged_vis, img_name, "intersection_05_merged")
        
        # Create grid from intersections
        spots = []
        h_sorted = sorted(h_lines, key=lambda l: (l[1] + l[3]) / 2)
        v_sorted = sorted(v_lines, key=lambda l: (l[0] + l[2]) / 2)
        
        for i in range(len(h_sorted) - 1):
            y1 = (h_sorted[i][1] + h_sorted[i][3]) / 2
            y2 = (h_sorted[i+1][1] + h_sorted[i+1][3]) / 2
            
            for j in range(len(v_sorted) - 1):
                x1 = (v_sorted[j][0] + v_sorted[j][2]) / 2
                x2 = (v_sorted[j+1][0] + v_sorted[j+1][2]) / 2
                
                width = abs(x2 - x1)
                height = abs(y2 - y1)
                
                # Check dimensions
                if 25 < width < 250 and 40 < height < 350:
                    x1_pad = max(0, int(x1) + 3)
                    y1_pad = max(0, int(y1) + 3)
                    x2_pad = min(img.shape[1], int(x2) - 3)
                    y2_pad = min(img.shape[0], int(y2) - 3)
                    
                    if x2_pad > x1_pad and y2_pad > y1_pad:
                        spots.append((x1_pad, y1_pad, x2_pad, y2_pad, None))
        
        self.log(f"Detected {len(spots)} parking spots from grid")
        return spots
    
    # ============================================================
    # OCCUPANCY ANALYSIS
    # ============================================================
    
    def check_occupancy(self, spot: Tuple, img: np.ndarray) -> bool:
        """Check if a parking spot is occupied using vehicle mask."""
        x1, y1, x2, y2 = spot[:4]
        
        # Create ROI mask
        mask_roi = np.zeros_like(self.vehicle_mask)
        cv2.rectangle(mask_roi, (x1, y1), (x2, y2), 255, -1)
        
        # Calculate intersection
        intersection = cv2.bitwise_and(self.vehicle_mask, mask_roi)
        occupied_pixels = cv2.countNonZero(intersection)
        total_pixels = (x2 - x1) * (y2 - y1)
        
        if total_pixels == 0:
            return False
        
        occupancy_ratio = occupied_pixels / total_pixels
        return occupancy_ratio > self.occupancy_threshold
    
    # ============================================================
    # MAIN PROCESSING PIPELINE
    # ============================================================
    
    def process_image(self, image_path: str):
        """Main processing pipeline - tries both methods."""
        print(f"\n{'='*60}")
        print(f"Processing: {os.path.basename(image_path)}")
        print(f"{'='*60}")
        
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not load {image_path}")
            return
        
        # Step 1: Generate vehicle mask (used by both methods)
        self.vehicle_mask = self.get_vehicle_mask(img)
        self.save_debug(self.vehicle_mask, image_path, "00_vehicle_mask")
        
        # Step 2: Try perimeter method first
        spots = []
        if self.use_perimeter_method:
            spots = self.detect_spots_perimeter(img, image_path)
            if len(spots) >= self.min_spots_threshold:
                self.method_used = "Perimeter"
                self.log(f"Using perimeter method ({len(spots)} spots)")
        
        # Step 3: Fallback to intersection method if needed
        if len(spots) < self.min_spots_threshold and self.use_intersection_method:
            self.log("Perimeter method found too few spots, trying intersection method...")
            spots = self.detect_spots_intersection(img, image_path)
            self.method_used = "Intersection"
        
        if not spots:
            print("No parking spots detected!")
            return
        
        # Step 4: Classify occupancy
        self.log("Classifying occupancy...")
        occupied_spots = []
        empty_spots = []
        
        for spot in spots:
            if self.check_occupancy(spot, img):
                occupied_spots.append(spot)
            else:
                empty_spots.append(spot)
        
        # Step 5: Visualize results
        vis_img = img.copy()
        
        for spot in empty_spots:
            x1, y1, x2, y2 = spot[:4]
            if spot[4] is not None:  # Perimeter method - draw rotated rect
                box = cv2.boxPoints(spot[4])
                box = np.int32(box)
                cv2.drawContours(vis_img, [box], 0, (0, 255, 0), 2)
            else:  # Intersection method - draw rectangle
                cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        for spot in occupied_spots:
            x1, y1, x2, y2 = spot[:4]
            if spot[4] is not None:
                box = cv2.boxPoints(spot[4])
                box = np.int32(box)
                cv2.drawContours(vis_img, [box], 0, (0, 0, 255), 2)
            else:
                cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        # Add statistics panel
        total = len(spots)
        occupied = len(occupied_spots)
        empty = len(empty_spots)
        
        panel_height = 120
        panel = np.zeros((panel_height, vis_img.shape[1], 3), dtype=np.uint8)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(panel, f"Method: {self.method_used}", (20, 25), 
                    font, 0.7, (255, 255, 255), 2)
        cv2.putText(panel, f"Total Spots: {total}", (20, 55), 
                    font, 0.7, (255, 255, 255), 2)
        cv2.putText(panel, f"Occupied: {occupied}", (20, 85), 
                    font, 0.7, (0, 0, 255), 2)
        cv2.putText(panel, f"Empty: {empty}", (20, 115), 
                    font, 0.7, (0, 255, 0), 2)
        
        result = np.vstack([panel, vis_img])
        
        # Save final result
        output_file = f"FINAL_{os.path.basename(image_path)}"
        output_path = os.path.join(self.output_dir, output_file)
        cv2.imwrite(output_path, result)
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"RESULTS")
        print(f"{'='*60}")
        print(f"  Method used:         {self.method_used}")
        print(f"  Total parking spots: {total}")
        print(f"  Occupied spots:      {occupied}")
        print(f"  Empty spots:         {empty}")
        print(f"  Occupancy rate:      {occupied/total*100:.1f}%")
        print(f"  Output saved to:     {output_path}")
        print(f"{'='*60}\n")


def main():
    """Process all parking lot images."""
    
    # Initialize detector with debug mode
    detector = HybridParkingDetector(debug=True)
    
    # Configure which methods to use
    detector.use_perimeter_method = True
    detector.use_intersection_method = True
    detector.min_spots_threshold = 5
    
    # Find input images
    input_dir = './input_images/'
    
    if not os.path.exists(input_dir):
        print(f"Error: Input directory {input_dir} not found!")
        return
    
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    
    for file in os.listdir(input_dir):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_files.append(file)
    
    if not image_files:
        print(f"No image files found in {input_dir}")
        return
    
    print(f"\n{'='*60}")
    print(f"HYBRID PARKING DETECTION SYSTEM")
    print(f"{'='*60}")
    print(f"Found {len(image_files)} images to process\n")
    
    # Process each image
    for img_file in sorted(image_files):
        input_path = os.path.join(input_dir, img_file)
        detector.process_image(input_path)
    
    print(f"\n{'='*60}")
    print("ALL PROCESSING COMPLETE!")
    print(f"Debug images and results saved to: {detector.output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()