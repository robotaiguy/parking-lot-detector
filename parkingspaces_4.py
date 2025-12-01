import cv2
import numpy as np
import os

class ParkingSystemPerimeter:
    def __init__(self):
        # CHANGE THIS TO YOUR OUTPUT PATH
        self.output_dir = "/mnt/c/wsl_outputs/"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # --- TUNING ---
        self.tophat_size = 25 
        
        # 1. LINE FILTER (The "Stick Sieve")
        # Strict filters to ensure we ONLY keep painted lines, not car edges.
        self.min_line_aspect = 3.0    # Must be very long/thin (Parking line)
        self.min_line_solidity = 0.85 # Must be solid (Not a squiggly tire track)
        
        # 2. SPOT FILTER (The "Room Sieve")
        self.min_area = 800           # Increased to ignore small road dashes
        self.max_area = 25000    
        self.min_aspect = 1.0         # Squares are okay (angled view)
        self.max_aspect = 5.0         # Too long = driving aisle, not a spot
        self.min_solidity = 0.80      # Spots are rectangular

    def save_debug(self, img, name, label):
        filename = f"{os.path.splitext(os.path.basename(name))[0]}_{label}.jpg"
        cv2.imwrite(os.path.join(self.output_dir, filename), img)

    def get_vehicle_mask(self, img):
        """Generates a mask of likely vehicles to check occupancy."""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 1. Saturation (Colorful cars)
        _, sat_mask = cv2.threshold(hsv[:,:,1], 40, 255, cv2.THRESH_BINARY)
        
        # 2. Dark Regions (Dark cars/Shadows)
        _, dark_mask = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
        
        # 3. Texture/Edges (Windshields, Grilles)
        edges = cv2.Canny(gray, 50, 150)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        edge_blob = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        mask = cv2.bitwise_or(sat_mask, dark_mask)
        mask = cv2.bitwise_or(mask, edge_blob)
        
        # Clean noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        return mask

    def process(self, image_path):
        print(f"Processing {image_path}...")
        img = cv2.imread(image_path)
        if img is None: return

        vis = img.copy()
        
        # --- STEP 1: VEHICLE MASK (For classification only) ---
        vehicle_mask = self.get_vehicle_mask(img)
        self.save_debug(vehicle_mask, image_path, "01_vehicle_mask")

        # --- STEP 2: LINE ISOLATION ---
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # CLAHE to handle shadows
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # TopHat to extract light, narrow features (lines)
        kernel_tophat = cv2.getStructuringElement(cv2.MORPH_RECT, (self.tophat_size, self.tophat_size))
        tophat = cv2.morphologyEx(enhanced, cv2.MORPH_TOPHAT, kernel_tophat)
        
        # Adaptive Threshold
        thresh = cv2.adaptiveThreshold(tophat, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 35, -5)
        self.save_debug(thresh, image_path, "02_raw_threshold")

        # --- STEP 3: STICK SIEVE (Remove Car Edges/Tire Tracks) ---
        # We only keep blobs that are mathematically "Sticks"
        num, labels, stats, _ = cv2.connectedComponentsWithStats(thresh, connectivity=8)
        clean_lines = np.zeros_like(thresh)
        
        # Analyze components using contours for RotatedRect access
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for c in contours:
            area = cv2.contourArea(c)
            if area < 50: continue # Ignore dust
            
            hull = cv2.convexHull(c)
            solidity = area / cv2.contourArea(hull) if cv2.contourArea(hull) > 0 else 0
            
            rect = cv2.minAreaRect(c)
            w, h = rect[1]
            short, long_s = min(w, h), max(w, h)
            aspect = long_s / short if short > 0 else 0
            
            # KEEP ONLY VALID PARKING LINES
            if aspect > self.min_line_aspect and solidity > self.min_line_solidity:
                cv2.drawContours(clean_lines, [c], -1, 255, -1)
                
        self.save_debug(clean_lines, image_path, "03_clean_lines")

        # --- STEP 4: GRID RECONSTRUCTION (Bridging Gaps) ---
        # Dilate lines to connect them into a grid. 
        # This bridges the gap under cars.
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(clean_lines, dilate_kernel, iterations=3)
        
        # Close to seal the "rooms" (spots)
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, close_kernel, iterations=3)
        self.save_debug(closed, image_path, "04_grid_closed")

        # --- STEP 5: SPOT DETECTION & CLASSIFICATION ---
        # Invert: Lines become walls, Background becomes Spots
        inverted = cv2.bitwise_not(closed)
        contours, _ = cv2.findContours(inverted, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        counts = {"EMPTY": 0, "OCCUPIED": 0}
        
        for c in contours:
            # 1. GEOMETRIC VALIDATION (Is this a parking spot?)
            area = cv2.contourArea(c)
            if area < self.min_area or area > self.max_area: continue

            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)
            if hull_area == 0: continue
            solidity = float(area) / hull_area
            if solidity < self.min_solidity: continue

            # Get Best-Fit Parallelogram
            rect = cv2.minAreaRect(c)
            (cx, cy), (w, h), angle = rect
            
            short = min(w, h)
            long_s = max(w, h)
            if short == 0: continue
            ar = long_s / short
            if not (self.min_aspect < ar < self.max_aspect): continue
            
            # 2. OCCUPANCY CHECK (Mask Intersection)
            # How much of this spot is covered by "Vehicle-like" pixels?
            mask_roi = np.zeros_like(vehicle_mask)
            cv2.drawContours(mask_roi, [c], -1, 255, -1)
            
            intersection = cv2.bitwise_and(vehicle_mask, mask_roi)
            occupied_pixels = cv2.countNonZero(intersection)
            occupancy_ratio = occupied_pixels / area
            
            if occupancy_ratio > 0.15: # 15% coverage = Occupied
                status = "OCCUPIED"
                color = (0, 0, 255) # Red
            else:
                status = "EMPTY"
                color = (0, 255, 0) # Green
                
            counts[status] += 1

            # 3. DRAW PERIMETER (Parallelogram)
            box = cv2.boxPoints(rect)
            box = np.int32(box)
            cv2.drawContours(vis, [box], 0, color, 2)

        self.draw_ui(vis, counts)
        self.save_debug(vis, image_path, "FINAL_PERIMETER")
        print(f"Done {image_path} | {counts}")

    def draw_ui(self, img, counts):
        cv2.rectangle(img, (20, 20), (300, 130), (0, 0, 0), -1)
        cv2.rectangle(img, (20, 20), (300, 130), (255, 255, 255), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, "PARKING STATUS", (40, 50), font, 0.7, (255, 255, 255), 2)
        cv2.putText(img, f"Empty: {counts['EMPTY']}", (40, 80), font, 0.6, (0, 255, 0), 1)
        cv2.putText(img, f"Occupied: {counts['OCCUPIED']}", (40, 105), font, 0.6, (0, 0, 255), 1)

if __name__ == "__main__":
    processor = ParkingSystemPerimeter()
    input_folder = "input_images"
    if os.path.exists(input_folder):
        images = [os.path.join(input_folder, f) for f in os.listdir(input_folder) 
                  if f.lower().endswith(('.jpg', '.png'))]
        for img in images:
            processor.process(img)