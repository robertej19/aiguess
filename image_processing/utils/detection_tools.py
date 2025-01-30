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
