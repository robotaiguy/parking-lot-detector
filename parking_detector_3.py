import cv2
import numpy as np
import os
import glob
from pathlib import Path

class ParkingLotDetector:
    def __init__(self, input_dir, output_dir):
        self.input_dir = input_dir
        self.output_dir = output_dir
        
        # Ensure output directory exists
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Heuristic Parameters (Tunable)
        self.car_aspect_ratio_range = (0.2, 3.0) # Cars/Spots are roughly rectangular
        self.min_spot_area = 750  # Min pixels to constitute a spot (adjust based on resolution)
        self.max_spot_area = 15000 # Max pixels
        
    def save_debug(self, filename, image, subfolder="debug"):
        """Saves intermediate images for troubleshooting."""
        path = os.path.join(self.output_dir, subfolder)
        if not os.path.exists(path):
            os.makedirs(path)
        cv2.imwrite(os.path.join(path, filename), image)

    def preprocess_image(self, img):
        """
        Enhance contrast to handle 'white-washed' asphalt.
        """
        # Convert to LAB color space to enhance Luminance channel
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        
        # Merge back
        limg = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        return enhanced

    def detect_lines_mask(self, img, base_filename):
        """
        Isolates parking lines using Color (White/Yellow) and Edge detection.
        """
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # --- 1. Color Masking ---
        # Yellow Mask
        # Hue: 15-35 roughly for yellow
        lower_yellow = np.array([15, 60, 100])
        upper_yellow = np.array([35, 255, 255])
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        # White Mask
        # Low Saturation, High Value
        lower_white = np.array([0, 0, 230]) # Slightly adjusted lower bound for robustness
        upper_white = np.array([180, 40, 255])
        mask_white = cv2.inRange(hsv, lower_white, upper_white)
        
        combined_color_mask = cv2.bitwise_or(mask_yellow, mask_white)
        self.save_debug(f"{base_filename}_01_color_mask.jpg", combined_color_mask)

        # --- 1b. Filter "Fat" Blobs (Cars/Objects) ---
        # Find contours to analyze shape thickness
        contours, _ = cv2.findContours(combined_color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create a black mask to draw the "bad" blobs to remove
        blobs_to_remove_mask = np.zeros_like(combined_color_mask)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Ignore tiny noise, only care about things big enough to be cars or lines
            if area < 30: 
                continue

            # Check "thickness" using MinAreaRect
            # minAreaRect returns ((center_x, center_y), (width, height), angle)
            rect = cv2.minAreaRect(cnt)
            (center), (width, height), angle = rect
            
            # The 'width' and 'height' here are of the rotated rectangle.
            # The smaller of the two is the "thickness" of the blob.
            min_dim = min(width, height)
            
            # Heuristic: 
            # A parking line is usually 5-15 pixels wide depending on zoom.
            # A car or building structure is usually > 20-30 pixels wide.
            # We filter out anything with a thickness > 20 pixels.
            if min_dim > 40: 
                cv2.drawContours(blobs_to_remove_mask, [cnt], -1, 255, -1)

        # Subtract the fat blobs from the original mask
        # We perform Bitwise AND with the NOT of the removal mask
        cleaned_mask = cv2.bitwise_and(combined_color_mask, cv2.bitwise_not(blobs_to_remove_mask))
        self.save_debug(f"{base_filename}_01b_cleaned_mask.jpg", cleaned_mask)

        # Use the cleaned mask for the rest of the pipeline
        combined_color_mask = cleaned_mask

        # --- 2. Edge Detection (Structure) ---
        # Blur slightly to reduce noise from asphalt texture
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Canny Edges
        edges = cv2.Canny(blurred, 30, 150)
        self.save_debug(f"{base_filename}_02_edges.jpg", edges)
        
        # --- 3. Combine ---
        # Dilate color mask slightly to ensure we catch the bulk of lines
        kernel = np.ones((3,3), np.uint8)
        dilated_color = cv2.dilate(combined_color_mask, kernel, iterations=1)
        
        # Strategy: Use Adaptive Thresholding on Gray to find "Bright" things relative to local neighborhood
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 55, -20)
        self.save_debug(f"{base_filename}_03_adaptive_thresh.jpg", thresh)

        # Final Mask Logic: 
        # For now, let's return the adaptive threshold which captures faint lines well,
        # but you might want to switch this to 'dilated_color' if the threshold is too noisy.
        # Given we just cleaned the color mask significantly, let's try returning the
        # cleaned color mask OR the edges for a cleaner result, 
        # or stick to thresh if you prefer the texture-based approach.
        
        # For the specific task of cleaning up "white-washed" images, 
        # often the cleaned color mask is safer than adaptive thresh which triggers on texture.
        # Let's return the adaptive thresh for now as per previous logic, but feel free to 
        # swap this return to 'dilated_color' if adaptive thresh is too noisy.
        final_mask = thresh
        return final_mask

    def segment_spots(self, line_mask, base_filename):
        """
        Uses the line mask to infer 'empty spaces' which are the spots.
        """
        # 1. Morphological operations to close gaps in dashed lines
        # We want to create a "grid" where the holes are the spots.
        kernel_size = 5
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        
        # Close gaps (dashed lines)
        closed = cv2.morphologyEx(line_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Dilate the lines to make them thick separators
        dilated = cv2.dilate(closed, kernel, iterations=2)
        self.save_debug(f"{base_filename}_04_lines_dilated.jpg", dilated)
        
        # 2. Invert the image. Now 'lines' are black, 'spots' are white blobs.
        inverted = cv2.bitwise_not(dilated)
        self.save_debug(f"{base_filename}_05_inverted_for_contours.jpg", inverted)
        
        # 3. Find Contours (potential spots)
        contours, _ = cv2.findContours(inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        valid_spots = []
        debug_contours = cv2.cvtColor(line_mask, cv2.COLOR_GRAY2BGR)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            
            # Geometric Filtering
            if self.min_spot_area < area < self.max_spot_area:
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = float(w) / h
                
                # Check aspect ratio (spots are generally elongated, either vertical or horizontal)
                # Allow a wide range because of diagonal spots
                if 0.2 < aspect_ratio < 5.0:
                    # Convex Hull to smooth out irregularities
                    hull = cv2.convexHull(cnt)
                    valid_spots.append(hull)
                    cv2.drawContours(debug_contours, [hull], -1, (0, 255, 0), 2)
        
        self.save_debug(f"{base_filename}_06_detected_polygons.jpg", debug_contours)
        return valid_spots

    def analyze_occupancy(self, img, spot_contour, base_filename, idx):
        """
        Determines if a spot is Empty or Occupied.
        Logic:
        1. Extract ROI.
        2. Check for Green (Trees/Grass) -> Empty.
        3. Check for Activity (Edges/Variance) -> High = Occupied, Low = Empty.
        """
        # Create mask for this specific spot
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [spot_contour], -1, 255, -1)
        
        # Extract ROI
        x, y, w, h = cv2.boundingRect(spot_contour)
        roi = img[y:y+h, x:x+w]
        roi_mask = mask[y:y+h, x:x+w]
        
        if roi.size == 0: return "Empty"

        # --- Feature 1: Green Detection (Tree/Grass Filter) ---
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        # Green range
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv_roi, lower_green, upper_green)
        
        # Apply ROI mask to ignore black background
        green_pixels = cv2.countNonZero(cv2.bitwise_and(green_mask, green_mask, mask=roi_mask))
        total_pixels = cv2.countNonZero(roi_mask)
        
        if total_pixels == 0: return "Empty"
        
        green_ratio = green_pixels / total_pixels
        
        # If significant green, it's a tree or landscaping -> Empty
        if green_ratio > 0.15: 
            return "Empty"

        # --- Feature 2: Activity/Edge Density (Car vs Asphalt) ---
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Canny on ROI
        edges_roi = cv2.Canny(gray_roi, 50, 150)
        edge_pixels = cv2.countNonZero(cv2.bitwise_and(edges_roi, edges_roi, mask=roi_mask))
        edge_density = edge_pixels / total_pixels
        
        # Standard Deviation of pixel intensity (Cars have more variation than asphalt)
        mean, std_dev = cv2.meanStdDev(gray_roi, mask=roi_mask)
        std_dev_val = std_dev[0][0]

        # --- Classification Logic ---
        # Tuned thresholds based on typical satellite imagery
        # Cars usually have high edge density OR high variance (specular highlights)
        
        is_occupied = False
        
        if edge_density > 0.05 or std_dev_val > 25.0:
            is_occupied = True
            
        # --- Feature 3: Small Object Filter (People/Poles) ---
        # If occupied, check if the "occupied" part is just a small blob?
        # This is harder with Canny, but we can check if the edges are scattered or clustered.
        # For this assignment, the Green filter + Edge Density is usually sufficient.
        # A person typically doesn't trigger high edge density across the WHOLE spot like a car does.
        
        return "Occupied" if is_occupied else "Empty"

    def process_folder(self):
        # file extensions
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        files = []
        for ext in extensions:
            files.extend(glob.glob(os.path.join(self.input_dir, ext)))
            
        print(f"Found {len(files)} images in {self.input_dir}")

        for filepath in files:
            filename = os.path.basename(filepath)
            name_no_ext = os.path.splitext(filename)[0]
            print(f"Processing {filename}...")
            
            # Load
            original_img = cv2.imread(filepath)
            if original_img is None:
                print(f"Error reading {filepath}")
                continue

            # 1. Preprocess
            enhanced_img = self.preprocess_image(original_img)
            self.save_debug(f"{name_no_ext}_00_enhanced.jpg", enhanced_img)

            # 2. Detect Lines
            line_mask = self.detect_lines_mask(enhanced_img, name_no_ext)
            
            # 3. Segment Spots
            spots = self.segment_spots(line_mask, name_no_ext)
            
            # 4. Classify Occupancy & Draw
            total_spots = len(spots)
            empty_count = 0
            occupied_count = 0
            
            result_img = original_img.copy()
            
            for i, spot in enumerate(spots):
                status = self.analyze_occupancy(original_img, spot, name_no_ext, i)
                
                if status == "Occupied":
                    color = (0, 0, 255) # Red
                    occupied_count += 1
                else:
                    color = (0, 255, 0) # Green
                    empty_count += 1
                
                # Draw Polygon
                cv2.drawContours(result_img, [spot], -1, color, 2)
                
                # Optional: Draw bounding rect for cleaner look
                x, y, w, h = cv2.boundingRect(spot)
                # cv2.rectangle(result_img, (x, y), (x+w, y+h), color, 2)

            # 5. Add Text Stats
            font = cv2.FONT_HERSHEY_SIMPLEX
            # Draw a background rectangle for text readability
            cv2.rectangle(result_img, (0, 0), (350, 120), (0, 0, 0), -1)
            cv2.putText(result_img, f"Total: {total_spots}", (10, 40), font, 1, (255, 255, 255), 2)
            cv2.putText(result_img, f"Empty: {empty_count}", (10, 80), font, 1, (0, 255, 0), 2)
            cv2.putText(result_img, f"Occupied: {occupied_count}", (10, 120), font, 1, (0, 0, 255), 2)
            
            # Console Output
            print(f"--- Results for {filename} ---")
            print(f"Total Spots: {total_spots}")
            print(f"Empty: {empty_count}")
            print(f"Occupied: {occupied_count}")
            print("-------------------------------")

            # Save Final
            final_path = os.path.join(self.output_dir, f"result_{filename}")
            cv2.imwrite(final_path, result_img)
            print(f"Saved result to {final_path}")

if __name__ == "__main__":
    # Define paths as requested
    input_path = "./input_images/"
    output_path = "/mnt/c/wsl_outputs/"
    
    detector = ParkingLotDetector(input_path, output_path)
    detector.process_folder()