import cv2, os, math, glob
import numpy as np
import matplotlib.pyplot as plt
from skimage import feature
from sklearn.linear_model import LinearRegression
from skimage.transform import rotate,warp



def detect_sky(image, do_morphology=True):
    # Convert BGR to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # --------------------------------------------------
    # 1) Threshold for "blue sky"
    #    (Hue around 90-130, depending on the type of sky)
    # --------------------------------------------------
    lower_blue = np.array([90, 50, 50])   # lower bound for blue
    upper_blue = np.array([130, 255, 255])  # upper bound for blue
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    # --------------------------------------------------
    # 2) Threshold for "white clouds"
    #    (Low saturation, high value)
    #    Typically Hue is “don't care”, so we might do:
    # --------------------------------------------------
    lower_white = np.array([0, 0, 200])    # near white
    upper_white = np.array([179, 50, 255]) # upper limit for "white" in HSV
    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    # Combine the two
    sky_mask = cv2.bitwise_or(mask_blue, mask_white)

    # Optional: morphological ops to remove noise and fill small holes
    if do_morphology:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        sky_mask = cv2.morphologyEx(sky_mask, cv2.MORPH_CLOSE, kernel)
        sky_mask = cv2.morphologyEx(sky_mask, cv2.MORPH_OPEN, kernel)

    return sky_mask

def estimate_horizon_line_by_edges(image):
    edges = feature.canny(image.astype(float),sigma=1)

    edge_pixels = np.column_stack(np.where(edges))
    if edge_pixels.shape[0]>1:
        X = edge_pixels[:,1].reshape(-1,1)
        y = edge_pixels[:,0]
        model = LinearRegression()
        model.fit(X,y)
        slope = model.coef_[0]
        intercept = model.intercept_

        if slope and intercept:
            return slope,intercept
        else:
            return 0,image.shape[0]/2
        

           
def rectify_horizon(image, slope, intercept):
    angle = np.degrees(np.arctan(slope))
    rotated_image = rotate(image,angle,resize=True,preserve_range=True)
    ri = rotated_image.astype(np.uint8)
    ri = translate_vertical(ri,-1*intercept)
    return ri

def downsampler(image,scale_factor=2):
    h,w = image.shape
    nw = w//scale_factor
    nh = h//scale_factor
    downsampled_image = cv2.resize(image,(nw,nh),interpolation=cv2.INTER_NEAREST)
    return downsampled_image

def rotate_and_center_horizon(image, slope, intercept, upside_down=False):
    """
    Rotate so the horizon becomes level (0 degrees).
    Vertically pin the horizon pivot at the canvas center,
    and center the entire image horizontally.
    """
    # 1) Dimensions of input
    H, W = image.shape[:2]
    big_side = 2 * max(W, H)  # final canvas dimension (square)

    # 2) Compute the rotation angle to flatten the horizon
    angle_radians = -np.arctan(slope)
    cos_a = np.cos(angle_radians)
    sin_a = np.sin(angle_radians)

    # 3) Choose pivot on the horizon at x = W/2
    #    (so that after rotation the horizon is leveled
    #    and we can place this pivot vertically at the center)
    pivot_x = W / 2.0
    pivot_y = slope * pivot_x + intercept

    # 4) Build 3x3 rotation-around-pivot matrix
    M_rot = np.array([
        [cos_a, -sin_a, pivot_x - pivot_x*cos_a + pivot_y*sin_a],
        [sin_a,  cos_a, pivot_y - pivot_x*sin_a - pivot_y*cos_a],
        [0,      0,     1]
    ], dtype=np.float32)

    # -- Compute locations of the image's corners after rotation --
    corners = np.array([
        [0,   0, 1],
        [W,   0, 1],
        [W,   H, 1],
        [0,   H, 1]
    ], dtype=np.float32)
    corners_rotated = (M_rot @ corners.T).T

    # -- Find the bounding box of those rotated corners --
    min_x = np.min(corners_rotated[:, 0])
    max_x = np.max(corners_rotated[:, 0])
    # Center of bounding box in the *rotated* space (horiz only)
    center_x = 0.5 * (min_x + max_x)

    # -- Pivot location after rotation (to pin vertically) --
    pivot_original = np.array([pivot_x, pivot_y, 1.0], dtype=np.float32)
    pivot_rotated = M_rot @ pivot_original
    pivot_rot_x, pivot_rot_y, _ = pivot_rotated

    # 5) We want:
    #    - pivot_rot_y to end up at vertical center of final canvas
    #    - bounding box center (center_x) to be at horizontal center
    canvas_cx = big_side / 2.0
    canvas_cy = big_side / 2.0

    tx = canvas_cx - center_x        # shift bounding box center to middle horizontally
    ty = canvas_cy - pivot_rot_y     # shift pivot to middle vertically

    # Build translation matrix
    M_translate = np.array([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1]
    ], dtype=np.float32)

    # 6) Combine rotation + translation matrices
    M_final = M_translate @ M_rot
    M_affine = M_final[:2, :]  # Extract the 2x3 affine transformation

    # 7) Warp the image onto the big canvas
    output_image = cv2.warpAffine(
        image,
        M_affine,
        (big_side, big_side),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)  # Black border
    )

    # 8) Flip if the frame is upside down
    if upside_down:
        output_image = cv2.flip(output_image, 0)  # 0 => flip vertically

    return output_image

def extract_object_and_background_masks(frame):
    """
    Given a frame with a single object and uniform background,
    this function extracts binary masks for the object and background.

    Args:
        frame (numpy.ndarray): Input frame (BGR or grayscale).
    
    Returns:
        object_mask (numpy.ndarray): Binary mask where object pixels are 1, background is 0.
        background_mask (numpy.ndarray): Binary mask where background pixels are 1, object is 0.
    """
    # Convert to grayscale if needed
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame.copy()
    
    # Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Otsu's thresholding to separate object and background
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours (external only)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        raise ValueError("No contours found! Check the image or adjust preprocessing steps.")

    # Assume the largest contour corresponds to the object
    largest_contour = max(contours, key=cv2.contourArea)

    # Create empty masks
    object_mask = np.zeros_like(gray, dtype=np.uint8)
    background_mask = np.ones_like(gray, dtype=np.uint8) * 255

    # Fill the object contour in the mask
    cv2.drawContours(object_mask, [largest_contour], -1, (255), thickness=cv2.FILLED)
    cv2.drawContours(background_mask, [largest_contour], -1, (0), thickness=cv2.FILLED)

    return object_mask, background_mask



def get_min_max_hsv(frame, mask):
    """
    Given a BGR frame and a mask, return the min and max values of H, S, and V within the masked area.
    """
    # Convert BGR to HSV
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Apply the mask
    masked_hsv = cv2.bitwise_and(hsv_frame, hsv_frame, mask=mask)

    # Extract only non-zero pixels
    nonzero_pixels = masked_hsv[np.where(mask > 0)]


    if nonzero_pixels.size == 0:
        print("No non-zero pixels found in the masked area.")
        return {"min_h": None, "max_h": None, "min_s": None, "max_s": None, "min_v": None, "max_v": None}

    # Get min and max for each channel
    min_h, max_h = np.min(nonzero_pixels[:, 0]), np.max(nonzero_pixels[:, 0])
    min_s, max_s = np.min(nonzero_pixels[:, 1]), np.max(nonzero_pixels[:, 1])
    min_v, max_v = np.min(nonzero_pixels[:, 2]), np.max(nonzero_pixels[:, 2])

    # return as min max arrays
    return {"min_h": min_h, "max_h": max_h, "min_s": min_s, "max_s": max_s, "min_v": min_v, "max_v": max_v}

def extract_contour_region(frame, contour):
    """
    Extract and return only the pixels inside the given contour.
    """
    # Create a mask with the same size as the frame, initialized to zero (black)
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    
    # Fill the contour with white color in the mask
    cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
    
    # Use the mask to extract the region of interest from the frame
    extracted_region = cv2.bitwise_and(frame, frame, mask=mask)
    
    return extracted_region, mask


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

# ========================================================
# Segmentation Routine (using k-means clustering)
# ========================================================
def segment_two_colors(frame, k=2):
    """
    Given a frame that has two dominant color regions, segment it using k-means clustering.
    Assumes that one cluster is the object (smaller area) and the other is the background.
    
    Returns a dictionary with:
      - object_mask, background_mask: binary masks for each region.
      - object_color_lab, background_color_lab: cluster centers in LAB color space.
      - object_color_bgr, background_color_bgr: cluster centers converted to BGR.
      - object_area, background_area: pixel counts for each cluster.
      - object_bbox: bounding box (x, y, w, h) for the object (largest contour in object mask).
    """
    # Convert image to LAB color space (LAB tends to be perceptually uniform)
    lab_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    
    # Reshape into 2D array of pixels and convert to float32 for k-means.
    pixel_values = lab_frame.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)
    
    # Define k-means criteria and run clustering.
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    attempts = 10
    ret, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, attempts, cv2.KMEANS_RANDOM_CENTERS)
    
    # Reshape labels back to the image shape.
    labels = labels.flatten()
    segmented_img = labels.reshape(frame.shape[:2])
    
    # Determine the area (pixel count) of each cluster.
    unique, counts = np.unique(labels, return_counts=True)
    cluster_areas = dict(zip(unique, counts))
    
    # Assume that the background is the cluster with more pixels.
    background_index = max(cluster_areas, key=cluster_areas.get)
    object_index     = min(cluster_areas, key=cluster_areas.get)
    
    # Create binary masks.
    object_mask = np.uint8((segmented_img == object_index)) * 255
    background_mask = np.uint8((segmented_img == background_index)) * 255
    
    # Retrieve the cluster centers (in LAB).
    object_color_lab     = centers[object_index]
    background_color_lab = centers[background_index]
    
    # Convert LAB centers to BGR for easier visualization.
    object_color_bgr = cv2.cvtColor(np.uint8([[object_color_lab]]), cv2.COLOR_LAB2BGR)[0][0]
    background_color_bgr = cv2.cvtColor(np.uint8([[background_color_lab]]), cv2.COLOR_LAB2BGR)[0][0]
    
    # Compute a bounding box for the object based on its mask.
    contours, _ = cv2.findContours(object_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        object_bbox = cv2.boundingRect(largest_contour)
    else:
        object_bbox = None
    
    result = {
        'object_mask': object_mask,
        'background_mask': background_mask,
        'object_color_lab': object_color_lab,
        'background_color_lab': background_color_lab,
        'object_color_bgr': object_color_bgr,
        'background_color_bgr': background_color_bgr,
        'object_area': cluster_areas[object_index],
        'background_area': cluster_areas[background_index],
        'object_bbox': object_bbox
    }
    return result
