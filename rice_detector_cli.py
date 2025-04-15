import cv2
import numpy as np
import matplotlib.pyplot as plt


def rice_detection_combined(image_path):
    """
    Detect and count rice grains using a combination of watershed and contour methods.

    Parameters:
    -----------
    image_path : str
        Path to the input image file

    Returns:
    --------
    tuple
        (contour_count, watershed_count, total_count)
    """
    # ==========================================================================
    # STEP 1: IMAGE LOADING AND PREPROCESSING
    # ==========================================================================

    # Load the image
    img = cv2.imread(image_path)

    # Convert the image from BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Apply brightness and contrast adjustment
    brightness = 6
    contrast = 0.60
    img_contrast = cv2.addWeighted(img, contrast, np.zeros(img.shape, img.dtype), 0, brightness)

    # ==========================================================================
    # STEP 2: SEGMENTATION
    # ==========================================================================

    # Color segmentation
    _, thresh1 = cv2.threshold(img_contrast, 82, 230, cv2.THRESH_BINARY)

    # Convert to grayscale
    gray = cv2.cvtColor(thresh1, cv2.COLOR_RGB2GRAY)

    # Gray segmentation
    _, thresh2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)

    # ==========================================================================
    # STEP 3: NOISE REMOVAL
    # ==========================================================================

    # Create kernel for morphological operations
    kernel = np.ones((3, 3), np.uint8)

    # Apply opening operation to remove small noise
    opening_img = cv2.morphologyEx(thresh2, cv2.MORPH_OPEN, kernel, iterations=6)

    # Apply erosion to separate connected grains
    erode_img = cv2.morphologyEx(opening_img, cv2.MORPH_ERODE, kernel, iterations=2)

    # ==========================================================================
    # STEP 4: WATERSHED ALGORITHM
    # ==========================================================================

    # 4.1: Create sure background by dilating the eroded image
    sure_bg = cv2.morphologyEx(erode_img, cv2.MORPH_DILATE, kernel, iterations=7)

    # 4.2: Distance transform to find central regions of grains
    dist_transform = cv2.distanceTransform(erode_img, cv2.DIST_L2, 5)

    # 4.3: Threshold to determine sure foreground
    _, sure_fg = cv2.threshold(dist_transform, 0.4 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    # 4.4: Find unknown region (region between sure foreground and sure background)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # 4.5: Marker labeling for watershed
    _, markers = cv2.connectedComponents(sure_fg)

    # Add 1 to all markers so background isn't 0
    markers = markers + 1

    # Mark unknown region as 0
    markers[unknown == 255] = 0

    # 4.6: Apply watershed algorithm
    markers = cv2.watershed(img, markers)

    # ==========================================================================
    # STEP 5: COMBINED DETECTION METHOD
    # ==========================================================================

    # Create a combined result image
    combined_result = img.copy()

    # Draw watershed boundaries in green
    combined_result[markers == -1] = [0, 255, 0]

    # 5.1: Process watershed regions
    watershed_centroids = []
    watershed_count = 0

    # Get unique marker IDs (excluding background and boundaries)
    unique_markers = np.unique(markers)
    valid_markers = [m for m in unique_markers if m > 1]

    for marker_id in valid_markers:
        # Create mask for this marker
        mask = np.zeros_like(gray)
        mask[markers == marker_id] = 255

        # Find contour
        rice_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if rice_contours:
            # Draw contour in green
            cv2.drawContours(combined_result, rice_contours, -1, (0, 255, 0), 2)

            # Get centroid
            M = cv2.moments(rice_contours[0])
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                watershed_centroids.append((cX, cY))
                watershed_count += 1

                # Label the grain with its number
                cv2.putText(combined_result, str(watershed_count), (cX, cY),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 2)

    # 5.2: Process contours that don't overlap with watershed regions
    contour_only_count = 0

    # Find contours using Canny edge detection
    contours, _ = cv2.findContours(cv2.Canny(opening_img, 30, 200),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for contour in contours:
        # Create a mask for this contour
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [contour], -1, 255, -1)

        # Check if this contour overlaps significantly with any watershed region
        overlap = cv2.bitwise_and(mask, opening_img)

        # Calculate overlap ratio (what percentage of the contour overlaps with opening_img)
        overlap_ratio = np.sum(overlap) / np.sum(mask) if np.sum(mask) > 0 else 0

        # If less than 55% overlap, consider it a separate rice grain
        if overlap_ratio < 0.55:
            # Draw contour in green
            cv2.drawContours(combined_result, [contour], -1, (0, 255, 0), 2)

            # Get centroid
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                contour_only_count += 1

                # Label the grain with its number (continuing from watershed_count)
                cv2.putText(combined_result, str(watershed_count + contour_only_count),
                            (cX + 20, cY + 20), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)

    # Calculate total count of rice grains
    total_count = watershed_count + contour_only_count

    # ==========================================================================
    # STEP 6: VISUALIZATION OF PROCESSING STEPS
    # ==========================================================================

    # Create figure with 3x3 subplots
    plt.figure(figsize=(18, 12))

    # Row 1: Original image processing
    plt.subplot(3, 3, 1)
    plt.imshow(img)
    plt.title("STEP 1: Original Image", fontweight='bold')
    plt.axis('off')

    plt.subplot(3, 3, 2)
    plt.imshow(img_contrast)
    plt.title("STEP 2: Contrast Adjusted", fontweight='bold')
    plt.axis('off')

    plt.subplot(3, 3, 3)
    plt.imshow(thresh1)
    plt.title("STEP 3: Color Segmentation", fontweight='bold')
    plt.axis('off')

    # Row 2: Grayscale processing
    plt.subplot(3, 3, 4)
    plt.imshow(gray, cmap='gray')
    plt.title("STEP 4: Grayscale Conversion", fontweight='bold')
    plt.axis('off')

    plt.subplot(3, 3, 5)
    plt.imshow(thresh2, cmap='gray')
    plt.title("STEP 5: Binary Thresholding", fontweight='bold')
    plt.axis('off')

    plt.subplot(3, 3, 6)
    plt.imshow(erode_img, cmap='gray')
    plt.title("STEP 6: Morphological Operations", fontweight='bold')
    plt.axis('off')

    # Row 3: Watershed and results
    plt.subplot(3, 3, 7)
    # Use jet colormap for better visualization of distance transform
    plt.imshow(dist_transform, cmap='jet')
    plt.title("STEP 7: Distance Transform", fontweight='bold')
    plt.axis('off')

    plt.subplot(3, 3, 8)
    plt.imshow(sure_fg, cmap='gray')
    plt.title("STEP 8: Sure Foreground", fontweight='bold')
    plt.axis('off')

    plt.subplot(3, 3, 9)
    plt.imshow(combined_result)
    plt.title(f"FINAL RESULT: {total_count} Rice Grains", fontweight='bold', color='darkgreen')
    plt.axis('off')

    # Add a suptitle to the figure
    plt.suptitle("Rice Grain Detection Process", fontsize=16, fontweight='bold', y=0.99)

    # Adjust layout and show
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.show()

    return len(contours), watershed_count, total_count


if __name__ == '__main__':
    # Path to the input image (update with your image path)
    image_path = r"C:\Users\DELL\Downloads\Simple Rice Detection\img.png"

    # Display informative message
    print("\n" + "=" * 70)
    print(" " * 25 + "RICE GRAIN DETECTION")
    print("=" * 70)

    # Run the detection algorithm
    contour_count, watershed_count, combined_count = rice_detection_combined(image_path)

    # Display results
    print("\nDETECTION RESULTS:")
    print("-" * 70)
    print(f"▶ Number of rice grains (contour method)  : {contour_count}")
    print(f"▶ Number of rice grains (watershed method): {watershed_count}")
    print(f"▶ TOTAL rice grains (combined method)     : {combined_count}")
    print("-" * 70 + "\n")