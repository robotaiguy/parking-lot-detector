import cv2
import os
import numpy as np
from shapely.geometry import Polygon

class ParkingDetails:
    def __init__(self):
        """
        Define heuristic parameters for parking spot detection and classification.
        """
        self.min_spot_area = 500
        self.max_spot_area = 15000
        self.binary_threshold = 180
        
    def detect_occupancy(self, warped_spot):
        """
        Determines if a spot is empty, occupied by a car, or a pedestrian
        using classical edge density and pixel luminance intensity.
        """
        # Convert parking space region to grayscale
        gray = cv2.cvtColor(warped_spot, cv2.COLOR_BGR2GRAY)
        
        # Apply edge detection (Canny) to find parking space boundaries and objects
        edges = cv2.Canny(gray, 50, 150)
        
        # Count non-zero edge pixels (edge density)
        edge_pixels = cv2.countNonZero(edges)
        total_pixels = warped_spot.shape[0] * warped_spot.shape[1]
        edge_density = edge_pixels / total_pixels
        
        # Simple blob detection for pedestrians (small distinct blobs)
        # Thresholding to find dark/light objects distinct from pavement
        _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_blob_area = 0
        if contours:
            max_blob = max([cv2.contourArea(c) for c in contours])
        blob_ratio = max_blob / total_pixels if total_pixels > 0 else 0

        # High edge density usually means a car (grilles, windows, complex shapes)
        # Fine tune these thresholds based on empirical testing
        if edge_density > 0.10:
            return "OCCUPIED", (0, 0, 255) # Red
        # Moderate blob but low edges = Smooth object (Pedestrian/covered object)
        elif blob_ratio > 0.15:
            return "PEDESTRIAN", (0, 165, 255) # Orange 
        # Low edges, low blob ratio means empty pavement
        else:
            return "EMPTY", (0, 255, 0) # Green

    def process_image(self, image_path):
        # Load image
        img = cv2.imread(image_path)
        output = img.copy()
        
        # Pre-processing for line detection (convert to grayscale and blur)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Adaptive Threshold to handle shadows/lighting changes
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 19, 5)

        # Finding the parking space lines by using contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Define list to hold detected parking spots
        spots = []
        
        # Filter contours that look like parking lines (long and thin)
        # Then infer the 'box' between two lines.
        # Note: For this assignment, I'm simplifying by detecting the 'box' formed 
        # by the parking lines themselves if they are closed loops, 
        # or estimating based on line spacing.
        
        # A more robust classical method: 'findContours' on the empty space
        # Dilate lines to close gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
        dilated = cv2.dilate(thresh, kernel, iterations=2)
        
        # Find Connected Components (potential spots) and inverting so the pavement becomes the "object"
        inverted_thresh = cv2.bitwise_not(dilated)
        spot_contours, _ = cv2.findContours(inverted_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Initialize counts
        spot_counts = {"EMPTY": 0, "OCCUPIED": 0, "PEDESTRIAN": 0}

        # Analyze each detected spot contour
        for c in spot_contours:
            # Get Oriented Bounding Box 
            rect = cv2.minAreaRect(c) 
            (x, y), (w, h), angle = rect
            area = w * h
            
            # Filter noise: Check if it's the size of a parking spot
            # These constraints might need tweaking per image resolution
            if self.min_spot_area < area < self.max_spot_area:
                # Aspect ratio check (spots are usually rectangular, not square)
                aspect_ratio = max(w, h) / min(w, h)
                if aspect_ratio < 1.5 or aspect_ratio > 6.0:
                    continue

                # Create the box points
                box = cv2.boxPoints(rect)
                box = np.int8(box)

                # Extract ROI for classification
                # We force the destination to be a standard size (e.g., 50x100)
                # This prevents squishing and ensures consistent pixel counts
                target_width, target_height = 50, 100
                
                src_pts = box.astype("float32")
                
                # Order points: top-left, top-right, bottom-right, bottom-left
                # (Re-using your sorting logic logic, simplified here)
                s = src_pts.sum(axis=1)
                diff = np.diff(src_pts, axis=1)
                ordered_pts = np.zeros((4, 2), dtype="float32")
                ordered_pts[0] = src_pts[np.argmin(s)]      # TL
                ordered_pts[2] = src_pts[np.argmax(s)]      # BR
                ordered_pts[1] = src_pts[np.argmin(diff)]   # TR
                ordered_pts[3] = src_pts[np.argmax(diff)]   # BL
                
                # Destination points are always a vertical rectangle
                dst_pts = np.array([
                    [0, 0],
                    [target_width - 1, 0],
                    [target_width - 1, target_height - 1],
                    [0, target_height - 1]], dtype="float32")

                # Get transform and Warp
                M = cv2.getPerspectiveTransform(ordered_pts, dst_pts)
                warped = cv2.warpPerspective(img, M, (target_width, target_height))

                # Determine occupancy and count
                status, color = self.detect_occupancy(warped)
                spot_counts[status] += 1

                # Draw the box with appropriate color
                cv2.drawContours(output, [box], 0, color, 2)

        # Render labeling overlay
        cv2.rectangle(output, (20, 20), (300, 140), (0,0,0), -1)
        y = 50
        for k, v in spot_counts.items():
            cv2.putText(output, f"{k}: {v}", (30, y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
            y += 25
            
        cv2.imshow("Result", output)
        cv2.waitKey(0)

if __name__ == "__main__":
    # Test each of the input images in 'input_images' folder
    for filename in os.listdir("./input_images"):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join("input_images", filename)
            print(f"Processing {image_path}...")
            detector = ParkingDetails()
            detector.process_image(image_path)
