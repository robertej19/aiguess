import cv2
import numpy as np

def frame_differencing(background, current_frame, diff_threshold=30):
    """
    Perform frame differencing between a static background and a current frame.
    
    Parameters:
        background (numpy.ndarray): The background image (BGR format).
        current_frame (numpy.ndarray): The current image (BGR format).
        diff_threshold (int): Threshold value for the difference image.
        
    Returns:
        numpy.ndarray: A binary image (mask) highlighting the differences.
    """
    # Convert both images to grayscale.
    bg_gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    
    # Compute the absolute difference between the background and current frame.
    diff = cv2.absdiff(bg_gray, curr_gray)
    
    # Threshold the difference image to create a binary image.
    # Pixels with differences greater than diff_threshold become white.
    _, diff_thresh = cv2.threshold(diff, diff_threshold, 255, cv2.THRESH_BINARY)
    
    # Optional: Apply morphological operations to reduce noise.
    kernel = np.ones((3, 3), np.uint8)
    # Remove small noise (opening).
    diff_thresh = cv2.morphologyEx(diff_thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    # Optionally, dilate the differences to make them more visible.
    diff_thresh = cv2.dilate(diff_thresh, kernel, iterations=1)
    
    return diff_thresh

# --- Example usage ---
if __name__ == "__main__":
    # Load the static background and current frame images.
    # For demonstration, replace 'background.jpg' and 'current.jpg' with your image paths.
    background = cv2.imread('background.jpg')
    current_frame = cv2.imread('current.jpg')
    
    if background is None or current_frame is None:
        print("Error: Could not load images.")
    else:
        # Obtain the binary difference mask.
        diff_mask = frame_differencing(background, current_frame, diff_threshold=30)
        
        # Display the result.
        cv2.imshow('Difference Mask', diff_mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
