import cv2
import numpy as np
import gradio as gr


def rice_detection_combined(img):
    """
    Detects and counts rice grains using a combination of image processing techniques.

    This function uses a hybrid approach combining watershed algorithm and contour detection
    to accurately identify individual rice grains even when they're touching.

    Args:
        img: Input image containing rice grains (numpy array)

    Returns:
        Combined result image with rice grains identified, counted and marked
    """
    # ----- PRE-PROCESSING PHASE -----
    # Enhance image by adjusting contrast and brightness for better segmentation
    brightness = 6
    contrast = 0.60
    img_contrast = cv2.addWeighted(img, contrast, np.zeros(img.shape, img.dtype), 0, brightness)

    # Initial color-based segmentation to separate rice from background
    _, thresh1 = cv2.threshold(img_contrast, 82, 230, cv2.THRESH_BINARY)

    # Convert to grayscale for further processing
    gray = cv2.cvtColor(thresh1, cv2.COLOR_RGB2GRAY)

    # Additional grayscale segmentation for better rice grain definition
    _, thresh2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)

    # ----- NOISE REMOVAL PHASE -----
    # Define morphological kernel for noise removal operations
    kernel = np.ones((3, 3), np.uint8)

    # Apply opening to remove small noise and smoothen grain boundaries
    opening_img = cv2.morphologyEx(thresh2, cv2.MORPH_OPEN, kernel, iterations=6)

    # Apply erosion to separate touching grains
    erode_img = cv2.morphologyEx(opening_img, cv2.MORPH_ERODE, kernel, iterations=2)

    # ----- WATERSHED SEGMENTATION PHASE -----
    # Step 1: Create sure background area by dilating the eroded image
    sure_bg = cv2.morphologyEx(erode_img, cv2.MORPH_DILATE, kernel, iterations=7)

    # Step 2: Distance transform to find the centers of foreground objects
    dist_transform = cv2.distanceTransform(erode_img, cv2.DIST_L2, 5)

    # Step 3: Threshold to get "sure foreground" areas (central regions of grains)
    _, sure_fg = cv2.threshold(dist_transform, 0.4 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    # Step 4: Find unknown region (area that's not sure background or sure foreground)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Step 5: Marker labeling for watershed algorithm
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1  # Shift all labels up by one so background isn't 0
    markers[unknown == 255] = 0  # Mark unknown region with zero

    # Step 6: Apply watershed algorithm to segment touching grains
    markers = cv2.watershed(img, markers)

    # ----- RESULTS COMBINATION PHASE -----
    # Create output image by copying the input
    combined_result = img.copy()
    combined_result[markers == -1] = [0, 255, 0]  # Mark watershed boundaries in green

    # ----- GRAIN COUNTING PHASE -----
    # Initialize grain counting variables
    watershed_centroids = []
    watershed_count = 0

    # Get all valid marker IDs (excluding background and boundaries)
    unique_markers = np.unique(markers)
    valid_markers = [m for m in unique_markers if m > 1]

    # Process each watershed-identified grain
    for marker_id in valid_markers:
        # Create binary mask for current grain
        mask = np.zeros_like(gray)
        mask[markers == marker_id] = 255

        # Find contour of the grain mask
        rice_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if rice_contours:
            # Draw grain contour in green
            cv2.drawContours(combined_result, rice_contours, -1, (0, 255, 0), 2)

            # Calculate and mark the centroid of the grain
            M = cv2.moments(rice_contours[0])
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                watershed_centroids.append((cX, cY))
                watershed_count += 1
                # Label each grain with its count number
                cv2.putText(combined_result, str(watershed_count), (cX, cY),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 2)

    # ----- SUPPLEMENTARY CONTOUR DETECTION PHASE -----
    # Find grains that watershed may have missed using contour detection
    contour_only_count = 0
    contours, _ = cv2.findContours(cv2.Canny(opening_img, 30, 200), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Process each contour that wasn't captured by watershed
    for contour in contours:
        # Create mask for current contour
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [contour], -1, 255, -1)

        # Check if this contour overlaps with any watershed region
        overlap = cv2.bitwise_and(mask, opening_img)
        overlap_ratio = np.sum(overlap) / np.sum(mask) if np.sum(mask) > 0 else 0

        # If less than 55% overlap, consider it a separate rice grain
        if overlap_ratio < 0.55:
            # Draw contour in green
            cv2.drawContours(combined_result, [contour], -1, (0, 255, 0), 2)

            # Calculate and mark the centroid
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                contour_only_count += 1
                # Label each additional grain with its count number
                cv2.putText(combined_result, str(watershed_count + contour_only_count),
                            (cX + 20, cY + 20), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)

    # ----- FINAL RESULT PREPARATION -----
    # Calculate total grain count from both methods
    total_count = watershed_count + contour_only_count

    # Get image dimensions
    height, width = combined_result.shape[:2]

    # Set text content
    text = f"Total Rice Grains: {total_count}"

    # Auto-scale font size based on image width
    font_scale = width / 1000  # Adjust this ratio as needed

    # Calculate text size to center it
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]
    text_x = (width - text_size[0]) // 2  # Center horizontally
    text_y = 50 + text_size[1]  # Position near top with padding

    # Add background for better readability
    text_bg_padding = 10
    text_bg_coords = (
        text_x - text_bg_padding, 
        text_y - text_size[1] - text_bg_padding,
        text_x + text_size[0] + text_bg_padding, 
        text_y + text_bg_padding
    )

    # Add total count annotation to the result image
    cv2.putText(combined_result, text,
                (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 
                font_scale, (255, 255, 255), 3)  # White text
    
    # Return the final annotated image with grain count
    return combined_result


# ----- GRADIO WEB INTERFACE -----
# Create the Gradio interface for user interaction
iface = gr.Interface(
    fn=rice_detection_combined,
    inputs=gr.Image(type="numpy"),  # Use gr.Image for input
    outputs=gr.Image(type="numpy"),  # Use gr.Image for output
    title="Rice Grain Detection and Counting System",
    description="Upload an image of rice grains to automatically detect and count them using advanced computer vision techniques."
)

# Launch the application when script is run directly
if __name__ == '__main__':
    iface.launch(share=True)  # Set share=True to create a public link
