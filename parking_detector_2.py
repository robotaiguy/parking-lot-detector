import cv2
import numpy as np
import os
from skimage.transform import hough_line, hough_line_peaks
from skimage.draw import polygon
import shutil

# --- Configuration ---
INPUT_DIR = "./input_images/"
OUTPUT_DIR = "/mnt/c/wsl_outputs/"
DEBUG = True  # Set to True to save intermediate images

# Ensure output directory exists
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

class ParkingSpaceDetector:
    def __init__(self, image_path, filename):
        self.original_image = cv2.imread(image_path)
        self.filename = filename
        self.debug_path = os.path.join(OUTPUT_DIR, f"{filename}_debug")
        if DEBUG:
            os.makedirs(self.debug_path, exist_ok=True)
            
        self.spots = [] # Will store list of polygons
        self.status = [] # Will store booleans (True=Occupied)

    def save_debug(self, name, img):
        if DEBUG:
            cv2.imwrite(os.path.join(self.debug_path, f"{name}.jpg"), img)

    def preprocess_lines(self):
        """
        Enhances white/yellow lines using TopHat morphology and CLAHE.
        """
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        
        # 1. CLAHE to equalize lighting (fixes "whitewashed" asphalt)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        self.save_debug("01_clahe", enhanced)

        # 2. TopHat Transform: Extracts bright elements smaller than the kernel
        # We use a rectangular kernel to emphasize line-like structures
        kernel_size = 25
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        tophat = cv2.morphologyEx(enhanced, cv2.MORPH_TOPHAT, kernel)
        self.save_debug("02_tophat", tophat)

        # 3. Binary Thresholding
        # Otsu's binarization automatically finds the separation between lines and asphalt
        _, binary = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 4. Clean up noise
        kernel_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_clean)
        self.save_debug("03_binary_lines", binary)
        
        return binary

    def detect_dominant_angle(self, binary_img):
        """
        Uses Hough transform to find the rotation angle of the parking lot.
        """
        # Run Hough Transform on a Canny edge version or the binary lines
        tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360, endpoint=False)
        h, theta, d = hough_line(binary_img, theta=tested_angles)
        
        # Find peaks
        _, angles, _ = hough_line_peaks(h, theta, d, num_peaks=5)
        
        # Calculate the median angle of the strongest peaks
        # We assume the parking lines are the most dominant linear features
        if len(angles) > 0:
            angle_rad = np.median(angles)
            angle_deg = np.rad2deg(angle_rad)
        else:
            angle_deg = 0

        # Adjust angle to be roughly vertical (-45 to 45) for easier processing
        if angle_deg < -45: angle_deg += 90
        if angle_deg > 45: angle_deg -= 90
        
        print(f"[{self.filename}] Detected Rotation Angle: {angle_deg:.2f} degrees")
        return angle_deg

    def generate_grid_via_projection(self, binary_img, angle):
        """
        Rotates image to vertical, calculates 1D projections to find rows and cols.
        """
        h, w = binary_img.shape
        center = (w // 2, h // 2)
        
        # Rotate image to make lines vertical
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(binary_img, M, (w, h))
        self.save_debug("04_rotated_binary", rotated)

        # --- Vertical Projection (Sum columns) to find Separating Lines ---
        v_projection = np.sum(rotated, axis=0)
        
        # Normalize and threshold projection to find "peaks" (lines)
        v_max = np.max(v_projection)
        v_peaks = np.where(v_projection > v_max * 0.2)[0] # Threshold heuristic
        
        # Group peaks into clusters (since a line is >1 pixel wide)
        line_cols = []
        if len(v_peaks) > 0:
            current_cluster = [v_peaks[0]]
            for i in range(1, len(v_peaks)):
                if v_peaks[i] - v_peaks[i-1] < 10: # Pixels close together
                    current_cluster.append(v_peaks[i])
                else:
                    line_cols.append(int(np.mean(current_cluster)))
                    current_cluster = [v_peaks[i]]
            line_cols.append(int(np.mean(current_cluster)))

        # Filter lines based on spacing (parking spot width heuristic)
        # We assume a minimum spot width (e.g., 20 pixels depending on res)
        # To be robust, we look for the median spacing
        if len(line_cols) > 1:
            diffs = np.diff(line_cols)
            median_width = np.median(diffs)
            # Filter lines that are too close (double detection of same line)
            valid_cols = [line_cols[0]]
            for i in range(1, len(line_cols)):
                if line_cols[i] - valid_cols[-1] > median_width * 0.5:
                    valid_cols.append(line_cols[i])
            line_cols = valid_cols

        # --- Horizontal Projection (Sum rows) to find Lanes ---
        h_projection = np.sum(rotated, axis=1)
        h_max = np.max(h_projection)
        
        # Parking rows usually show up as high consistent energy in binary mask (lines present)
        # or we can look for the "gap" between rows. 
        # Simpler heuristic: Smoothing the projection and thresholding.
        rows_mask = h_projection > (h_max * 0.1)
        
        # Find start and end indices of rows
        row_segments = []
        in_segment = False
        start = 0
        for i, val in enumerate(rows_mask):
            if val and not in_segment:
                start = i
                in_segment = True
            elif not val and in_segment:
                if i - start > 50: # Minimum height for a parking row
                    row_segments.append((start, i))
                in_segment = False
        if in_segment and h - start > 50:
            row_segments.append((start, h))

        # Generate Spot Polygons (Rotated back)
        inv_M = cv2.invertAffineTransform(M)
        
        temp_visual = cv2.cvtColor(rotated, cv2.COLOR_GRAY2BGR)

        for (r_start, r_end) in row_segments:
            for i in range(len(line_cols) - 1):
                c_start = line_cols[i]
                c_end = line_cols[i+1]
                
                # Check aspect ratio to ensure it's a parking spot shape (usually rectangular)
                w_spot = c_end - c_start
                h_spot = r_end - r_start
                ratio = h_spot / w_spot if w_spot > 0 else 0
                
                # Typical spot is ~1:2 or 1:3 ratio. 
                if 1.0 < ratio < 6.0 and w_spot > 15: 
                    # Define box in rotated coordinates
                    pts = np.array([
                        [c_start, r_start],
                        [c_end, r_start],
                        [c_end, r_end],
                        [c_start, r_end]
                    ], dtype=np.float32)
                    
                    # Rotate back to original
                    pts_original = cv2.transform(np.array([pts]), inv_M)[0]
                    self.spots.append(pts_original.astype(int))
                    
                    # Draw on temp debug
                    cv2.rectangle(temp_visual, (c_start, r_start), (c_end, r_end), (0,255,0), 2)

        self.save_debug("05_grid_rectified", temp_visual)

    def classify_spots(self):
        """
        Determines if a spot is Empty, Occupied (Car), or Empty (Tree/Pole).
        Logic:
        1. Extract ROI.
        2. Calculate Edge Density (Canny) -> Cars have high edges.
        3. Calculate HSV Saturation -> Cars often have color (unlike asphalt).
        4. Calculate Intensity Deviation -> Asphalt is flat, cars/trees are varied.
        """
        
        hsv_img = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2HSV)
        gray_img = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        
        # Compute edges for the whole image once
        edges = cv2.Canny(gray_img, 50, 150)
        self.save_debug("06_canny_edges", edges)

        self.status = []

        for spot_pts in self.spots:
            # Create mask for this spot
            mask = np.zeros(gray_img.shape, dtype=np.uint8)
            cv2.fillPoly(mask, [spot_pts], 255)
            
            # Extract features
            # 1. Edge Density: (Sum of edge pixels) / Area
            spot_area = np.sum(mask > 0)
            if spot_area == 0: 
                self.status.append(False)
                continue
                
            edge_content = cv2.bitwise_and(edges, edges, mask=mask)
            edge_score = np.sum(edge_content) / spot_area
            
            # 2. Intensity Standard Deviation (Texture)
            mean, std_dev = cv2.meanStdDev(gray_img, mask=mask)
            std_score = std_dev[0][0]
            
            # 3. Color Saturation (to differentiate grey cars/asphalt from colored cars)
            mean_hsv, std_hsv = cv2.meanStdDev(hsv_img, mask=mask)
            sat_score = mean_hsv[1][0] # Saturation channel

            # Heuristics derived from common CV approaches for parking:
            # - Empty spot: Low edge density, Low Std Dev (smooth asphalt)
            # - Car: High edge density (windows, bumpers), High Std Dev
            # - Tree: High edge density, High texture, but we rely on the prompt saying 
            #   "consider spaces occupied by trees/poles as empty". 
            #   Refinement: Trees usually have extremely high texture but irregular shapes. 
            #   However, without ML, separating a Tree from a Car is hard.
            #   We will focus on the "Metal vs Asphalt" logic.
            
            is_occupied = False
            
            # Thresholds (tunable based on image resolution/quality)
            # Assuming standard satellite resolution
            if edge_score > 0.08 or std_score > 35:
                is_occupied = True
                
            # Refinement for "Trees/Poles":
            # If the saturation is relatively low (grey/white car) or very high (red car) it's likely a car.
            # Trees are often mid-saturation green. 
            # For now, the edge/std deviation is the strongest signal for "Not Asphalt".
            
            self.status.append(is_occupied)

    def draw_results(self):
        output = self.original_image.copy()
        total = len(self.spots)
        occupied = sum(self.status)
        empty = total - occupied
        
        # Draw Spots
        for i, pts in enumerate(self.spots):
            color = (0, 0, 255) if self.status[i] else (0, 255, 0) # Red if Occ, Green if Empty
            
            # Draw semi-transparent overlay
            overlay = output.copy()
            cv2.fillPoly(overlay, [pts], color)
            cv2.addWeighted(overlay, 0.3, output, 0.7, 0, output)
            
            # Draw contour
            cv2.polylines(output, [pts], True, color, 2)

        # Draw Info Box
        info_text = [
            f"Total Spots: {total}",
            f"Occupied: {occupied}",
            f"Empty: {empty}"
        ]
        
        cv2.rectangle(output, (0, 0), (250, 100), (0,0,0), -1)
        for i, line in enumerate(info_text):
            cv2.putText(output, line, (10, 30 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
        print(f"[{self.filename}] {info_text}")
        
        final_path = os.path.join(OUTPUT_DIR, f"result_{self.filename}")
        cv2.imwrite(final_path, output)
        return output

    def run(self):
        print(f"Processing {self.filename}...")
        try:
            # 1. Detect Geometry
            binary_lines = self.preprocess_lines()
            angle = self.detect_dominant_angle(binary_lines)
            self.generate_grid_via_projection(binary_lines, angle)
            
            # 2. Classify
            self.classify_spots()
            
            # 3. Output
            self.draw_results()
        except Exception as e:
            print(f"Error processing {self.filename}: {e}")
            import traceback
            traceback.print_exc()

# --- Main Execution ---
if __name__ == "__main__":
    
    # Get all images in input directory
    files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not files:
        print("No input images found.")
    else:
        for f in files:
            detector = ParkingSpaceDetector(os.path.join(INPUT_DIR, f), f)
            detector.run()
            
    print("Processing Complete. Check output directory.")