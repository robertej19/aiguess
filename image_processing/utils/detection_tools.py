import cv2, os, math, glob, sys
import numpy as np
import matplotlib.pyplot as plt
from skimage import feature
from sklearn.linear_model import LinearRegression
from skimage.transform import rotate,warp
from itertools import product

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.common_tools import show_bgr

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

def get_min_max_lab_values(frame, mask):
    # Convert the frame to Lab color space
    lab_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    
    # Apply the mask to the Lab image (only select pixels where the mask is true)
    masked_lab = cv2.bitwise_and(lab_frame, lab_frame, mask=mask)
    
    # Reshape the masked frame to a 2D array (pixels x channels)
    masked_lab_reshaped = masked_lab.reshape(-1, 3)
    
    # Mask out the pixels that are not selected by the mask (i.e., mask == 0)
    masked_lab_reshaped = masked_lab_reshaped[np.all(masked_lab_reshaped != 0, axis=1)]
    
    # Calculate the min and max for each channel (L, A, B)
    min_vals = np.min(masked_lab_reshaped, axis=0)
    max_vals = np.max(masked_lab_reshaped, axis=0)
    
    return min_vals, max_vals


def plot_min_max_lab_colors(min_vals, max_vals):
    # Generate all combinations of min and max values for L, A, B
    combinations = list(product(*zip(min_vals, max_vals)))
    
    # Normalize and clip Lab values to the 0-255 range
    lab_colors = np.array(combinations, dtype=np.uint8)

    # Convert the Lab values to RGB (or BGR) using OpenCV
    bgr_colors = cv2.cvtColor(lab_colors.reshape(1, 8, 3), cv2.COLOR_LAB2RGB)

    # Plot the 8 colors in a row
    plt.figure(figsize=(18, 2))
    plt.imshow(bgr_colors)
    plt.axis('off')
    plt.title(f"Min (L, A, B): {min_vals} | Max (L, A, B): {max_vals}")
    plt.show()



def create_lab_range_mask(frame, min_vals, max_vals):
    # Convert the frame to Lab color space
    lab_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    
    # Create the mask using inRange, which checks each pixel's value within the min and max range for each channel
    # make it be uin64
    lower_bound = np.array(min_vals, dtype=np.uint64)
    upper_bound = np.array(max_vals, dtype=np.uint64)
    
    # Apply the range threshold for L, A, and B channels
    mask = cv2.inRange(lab_frame, lower_bound, upper_bound)
    
    return mask


def create_donut_mask_with_exclusion(frame, contour,
                                       outer_pad_ratio=0.5,
                                       exclusion_pad_ratio=3.0):
    """
    Creates a "donut" mask from an input contour, where the donut is defined
    as the region between an outer dilation of the contour and an inner
    exclusion region that is the original object grown by an additional padding.
    
    The inner exclusion region will include the original contour plus extra padding.
    For example, if the object's approximate width is 10 pixels and exclusion_pad_ratio
    is 3.0, then the inner exclusion region will roughly be 10 + (10*3) = 40 pixels wide.
    
    Parameters:
      frame              - Input image (only its size is used).
      contour            - The contour (numpy array, as returned by cv2.findContours).
      outer_pad_ratio    - The fraction of the contour bounding-box width to use
                           for the outer dilation (default is 0.5, i.e. 50%).
      exclusion_pad_ratio- The factor to multiply the bounding-box width by to get the
                           extra exclusion padding. (For example, 3.0 means 300% extra.)
    
    Returns:
      donut_mask         - A binary mask (np.uint8) where the donut region is 255 and
                           all other areas are 0.
    """
    # Create an empty mask for the object (filled contour).
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, [contour], -1, 255, thickness=-1)
    
    # Compute the bounding rectangle of the contour.
    x, y, w_box, h_box = cv2.boundingRect(contour)
    
    # Compute the padding amounts in pixels.
    # Outer pad: used for the outer boundary of the donut.
    outer_pad = int(w_box * outer_pad_ratio)
    # Exclusion pad: extra padding to be added to the original object.
    exclusion_pad = int(w_box * exclusion_pad_ratio)
    
    # --- Create Outer Mask ---
    # We want a donut that is relatively wide. Here, we choose to dilate the original object
    # by an amount corresponding to twice the outer_pad (this is arbitrary; adjust as needed).
    outer_dilation_radius = 2 * outer_pad
    outer_kernel_size = 2 * outer_dilation_radius + 1
    outer_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                             (outer_kernel_size, outer_kernel_size))
    outer_mask = cv2.dilate(filled, outer_kernel)
    
    # --- Create Inner Exclusion Mask ---
    # This mask excludes the object itself plus extra padding.
    inner_kernel_size = 2 * exclusion_pad + 1
    inner_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                             (inner_kernel_size, inner_kernel_size))
    inner_exclusion = cv2.dilate(filled, inner_kernel)
    
    # --- Create the Donut Mask ---
    # The donut is the region between the outer boundary and the inner exclusion region.
    donut_mask = cv2.subtract(outer_mask, inner_exclusion)
    
    return donut_mask