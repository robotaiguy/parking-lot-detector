import cv2
import numpy as np
import os

class ParkingSystem:
    def __init__(self):
        # CONFIGURATION
        # Standardize all spots to this size for analysis
        self.warp_w, self.warp_h = 60, 120 
        
        # Area filters (adjust if images are 4k vs 720p)
        # These defaults work for standard 1080p-ish drone shots
        self.min_area = 400
        self.max_area = 30000
        
        # Aspect ratio of a parking spot (usually 1:2)
        self.min_aspect = 1.2
        self.max_aspect = 5.0

    def order_points(self, pts):
        """
        Sorts coordinates to: top-left, top-right, bottom-right, bottom-left
        Necessary for correct perspective warping.
        """
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]   # Top-left
        rect[2] = pts[np.argmax(s)]   # Bottom-right
        
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # Top-right
        rect[3] = pts[np.argmax(diff)]  # Bottom-left
        return rect

    def get_spot_status(self, spot_img):
        """
        Analyzes a standardized 60x120 spot image.
        Returns: Status String, Color Tuple
        """
        # 1. Convert to grayscale
        gray = cv2.cvtColor(spot_img, cv2.COLOR_BGR2GRAY)
        
        # 2. Calculate Edge Density (Canny)
        # Cars have many internal edges (windows, lights, hood lines)
        # Empty spots are smooth concrete (few edges)
        edges = cv2.Canny(gray, 50, 200)
        edge_score = np.count_nonzero(edges) / (self.warp_w * self.warp_h)
        
        # 3. Calculate Texture/Variance (Laplacian)
        # Helps distinguish pedestrians (small blobs) from cars (full blobs)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()

        # --- HEURISTICS ---
        # NOTE: These thresholds are empirical. 
        
        # CASE: OCCUPIED (High edges, high texture)
        if edge_score > 0.12:
            return "OCCUPIED", (0, 0, 255) # Red
            
        # CASE: PEDESTRIAN / OCCLUDED 
        # (Lower edges than a car, but higher variance than empty cement)
        elif edge_score > 0.05 or variance > 300:
            return "OCCLUDED", (0, 165, 255) # Orange
            
        # CASE: EMPTY
        else:
            return "EMPTY", (0, 255, 0) # Green

    def process(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error loading {image_path}")
            return

        original = img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # --- PRE-PROCESSING ---
        # Blur to reduce noise (asphalt grain)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Adaptive Threshold to find lines/boxes despite shadows
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 25, 10)
        
        # Morphological operations to close gaps in painted lines
        # We dilate significantly to force the parking lines to form closed loops
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        # Dilate: make white lines thicker
        dilated = cv2.dilate(thresh, kernel, iterations=2) 
        # Close: fill small black holes inside white regions
        closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Find Contours
        contours, _ = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        stats = {"EMPTY": 0, "OCCUPIED": 0, "OCCLUDED": 0}
        
        valid_spots = 0

        for c in contours:
            # 1. Geometry Filter
            area = cv2.contourArea(c)
            if not (self.min_area < area < self.max_area):
                continue
            
            # Get oriented bounding box
            rect = cv2.minAreaRect(c)
            (x, y), (w, h), angle = rect
            
            # Handle OpenCV's dimension swapping (width isn't always shorter side)
            spot_w = min(w, h)
            spot_h = max(w, h)
            
            # Aspect Ratio Check (Spots are rectangles, 1:2 to 1:5 roughly)
            ar = spot_h / spot_w if spot_w > 0 else 0
            if not (self.min_aspect < ar < self.max_aspect):
                continue

            # 2. Perspective Warp (Normalize the spot)
            box = cv2.boxPoints(rect)
            box = np.int32(box)
            
            # Sort corners
            src_pts = box.astype("float32")
            src_pts = self.order_points(src_pts)
            
            # Destination is always a vertical rectangle
            dst_pts = np.array([
                [0, 0],
                [self.warp_w - 1, 0],
                [self.warp_w - 1, self.warp_h - 1],
                [0, self.warp_h - 1]], dtype="float32")
            
            # Warp
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            warped = cv2.warpPerspective(img, M, (self.warp_w, self.warp_h))
            
            # 3. Classify
            status, color = self.get_spot_status(warped)
            
            # Update stats & Draw
            stats[status] += 1
            valid_spots += 1
            
            # Draw the box
            cv2.drawContours(original, [box], 0, color, 2)
            
            # Draw small circle at center to visualize "center of mass"
            center_int = (int(x), int(y))
            cv2.circle(original, center_int, 2, (0,0,255), -1)

        # --- DRAW OVERLAY ---
        # Create a semi-transparent panel for text
        overlay = original.copy()
        cv2.rectangle(overlay, (20, 20), (320, 150), (0, 0, 0), -1)
        alpha = 0.6
        cv2.addWeighted(overlay, alpha, original, 1 - alpha, 0, original)

        # Text Metrics
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(original, f"Total Spots: {valid_spots}", (35, 50), font, 0.7, (255, 255, 255), 2)
        cv2.putText(original, f"Empty: {stats['EMPTY']}", (35, 80), font, 0.6, (0, 255, 0), 1)
        cv2.putText(original, f"Occupied: {stats['OCCUPIED']}", (35, 105), font, 0.6, (0, 0, 255), 1)
        cv2.putText(original, f"Pedestrian: {stats['OCCLUDED']}", (35, 130), font, 0.6, (0, 165, 255), 1)

        # Display
        scale = 0.8 # Resize for screen if images are huge
        display_h, display_w = int(original.shape[0]*scale), int(original.shape[1]*scale)
        resized = cv2.resize(original, (display_w, display_h))
        
        # Save to disk
        output_filename = "result_" + os.path.basename(image_path)
        cv2.imwrite(output_filename, resized)
        print(f"Saved result to: {output_filename}")

if __name__ == "__main__":
    # Create an instance and run
    # Ensure you have your images in the folder
    processor = ParkingSystem()
    
    # List of images to process
    # Replace these filenames with the actual paths to your downloaded images
    images = ["1.jpg", "2.jpg", "4.jpg", "5.jpg"]
    input_images = os.path.join('.', 'input_images')
    images = [os.path.join(input_images, img) for img in images]
    for img_name in images:
        if os.path.exists(img_name):
            print(f"Starting {img_name}...")
            processor.process(img_name)
        else:
            print(f"File not found: {img_name} - moving to next.")