import cv2
import numpy as np
import matplotlib.pyplot as plt

def parking_analysis_simple(image_path):
    # Load image for preprocessing
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not load image.")
        return

    # Converty image copy to grayscale
    output_img = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Blur to remove high frequency noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use Adaptive Thresholding to highlight the white/yellow lines against dark pavement
    # Adjust matrix size and C value based on image characteristics
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 25, 15
    )

    # Morphological operations to clean up noise and close gaps in lines
    # Adjust kernel size, iterations, and operations as needed
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # Dilate to make lines thicker and connect them
    dilated = cv2.dilate(thresh, kernel, iterations=2) 
    
    # Find contours (potential parking spots)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create an empty list to hold detected parking spots
    parking_spots = []
    
    # Since parking spaces are roughly oriented rectangular shapes and within size limits,
    # define criteria to filter contours
    MIN_AREA = 1500  # Tune this
    MAX_AREA = 10000 # Tune this
    MIN_ASPECT = 0.5
    MAX_ASPECT = 2.5

    # Iterate through contours to find matches for parking spot criteria
    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        if MIN_AREA < area < MAX_AREA:
            # Approximate the contour to a polygon
            peri = cv2.arcLength(cnt, True)
            poly = cv2.approxPolyDP(cnt, 0.04 * peri, True)
            
            # Get bounding box for aspect ratio check
            x, y, w, h = cv2.boundingRect(poly)
            aspect_ratio = float(w) / h
            
            # Ensure the candidate has roughly 4 corners and valid aspect ratio
            if len(poly) == 4 and MIN_ASPECT < aspect_ratio < MAX_ASPECT:
                parking_spots.append(poly)

    # Define variables to count occupied and empty spots
    total_spots = len(parking_spots)
    occupied_count = 0
    empty_count = 0

    # Create a threshold variable for variance to classify occupancy
    # Fine tune this based on lighting and image conditions
    VARIANCE_THRESHOLD = 45

    # Analyze each detected parking spot
    for spot in parking_spots:
        # Create a mask for each individual spot
        mask = np.zeros(gray.shape, np.uint8)
        cv2.drawContours(mask, [spot], -1, 255, -1)
        
        # Extract the mean standard deviation of luminance intensity in this spot
        # Using mean intensity alone is risky due to shadows.
        # Standard Deviation (Texture) is robust: cars are "noisy", pavement is "smooth"
        mean, std_dev = cv2.meanStdDev(gray, mask=mask)
        current_std_dev = std_dev[0][0]

        # Classify based on standard deviation
        if current_std_dev > VARIANCE_THRESHOLD:
            # Occupied -> Draw Red
            color = (0, 0, 255) 
            occupied_count += 1
            status = "Occupied"
        else:
            # Empty -> Draw Green
            color = (0, 255, 0)
            empty_count += 1
            status = "Empty"

        # Draw the polygon and label
        cv2.drawContours(output_img, [spot], -1, color, 2)
        
        # Find center for labeling
        M = cv2.moments(spot)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.putText(output_img, str(int(current_std_dev)), (cX - 20, cY), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Print the results summary
    print(f"Total Spots Detected: {total_spots}")
    print(f"Occupied: {occupied_count}")
    print(f"Empty: {empty_count}")

    # Show result using matplotlib to display in notebooks or environments without GUI support
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))
    plt.title(f"Total: {total_spots} | Occupied: {occupied_count} | Empty: {empty_count}")
    plt.axis('off')
    plt.show()

    # Save result
    cv2.imwrite('parking_1_result.jpg', output_img)

# For each image in 'input_images' folder, call the function
if __name__ == "__main__":
    import os
    input_folder = './input_images'
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_folder, filename)
            print(f"Processing {image_path}...")
            parking_analysis_simple(image_path)
