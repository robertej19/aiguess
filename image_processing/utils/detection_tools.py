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

def translate_vertical(image, distance):
    """
    Translates the given image vertically by the specified distance.

    Parameters:
        image (numpy.ndarray): The input image.
        distance (int): The number of pixels to move the image downwards.

    Returns:
        numpy.ndarray: The translated image.
    """
    rows, cols = image.shape[:2]
    
    # Define the translation matrix: no horizontal shift, vertical shift by 'distance' pixels
    M = np.float32([[1, 0, 0],
                    [0, 1, distance]])
    # Apply the affine transformation
    translated_image = cv2.warpAffine(image, M, (cols, rows))
    return 

def create_line_roi_mask_vectorized(width, height, slope, intercept, above=True):
    """
    Create a mask for y = slope*x + intercept using NumPy vectorization.
    If above=True => keep y < slope*x + intercept
    """
    # Create a grid of x,y coords
    X, Y = np.meshgrid(np.arange(width), np.arange(height))

    line_vals = slope*X + intercept
    if above:
        condition = (Y < line_vals)
    else:
        condition = (Y > line_vals)

    mask = np.zeros((height, width), dtype=np.uint8)
    mask[condition] = 255
    return mask

def rotate_and_center_horizon(image, slope, intercept):
    """
    Rotate 'image' so that the line y = slope*x + intercept becomes horizontal
    and is vertically centered in a fixed-size output canvas.
    
    The output canvas is a square with side = 2 * max(W, H), ensuring no cropping
    for any rotation angle.

    Parameters
    ----------
    image : np.ndarray
        Input image of shape (H, W).
    slope : float
        Slope (m) of horizon line in the original image (y = m*x + b).
    intercept : float
        Intercept (b) of the horizon line in the original image.

    Returns
    -------
    output_image : np.ndarray
        Rotated/translated image in a canvas of size:
          (2 * max(W, H), 2 * max(W, H)).
        The horizon line is horizontal and centered vertically.
    """

    # 1) Dimensions of input
    H, W = image.shape[:2]
    big_side = 2 * max(W, H)  # final canvas dimension for both width and height

    # 2) Compute the rotation angle to flatten the horizon
    # slope = tan(angle) => angle = arctan(slope). We negate it to make slope=0
    angle_radians = -np.arctan(slope)
    cos_a = np.cos(angle_radians)
    sin_a = np.sin(angle_radians)

    # 3) Choose pivot on horizon at x = W/2
    pivot_x = W / 2.0
    pivot_y = slope * pivot_x + intercept

    # 4) Build 3x3 rotation-around-pivot matrix
    #    Step A: Translate pivot to origin
    #    Step B: Rotate by angle_radians
    #    Step C: Translate back
    M_rot = np.array([
        [ cos_a, -sin_a,  pivot_x - pivot_x*cos_a + pivot_y*sin_a],
        [ sin_a,  cos_a,  pivot_y - pivot_x*sin_a - pivot_y*cos_a],
        [     0,      0,                                 1       ]
    ], dtype=np.float32)

    # 5) Find pivot location AFTER rotation
    pivot_original = np.array([pivot_x, pivot_y, 1.0], dtype=np.float32)
    pivot_rotated = M_rot @ pivot_original
    pivot_rot_x, pivot_rot_y, _ = pivot_rotated

    # 6) We want the pivot to be at (canvas_center_x, canvas_center_y)
    #    - The horizon is thus horizontally in the center (x-dimension can be center or anywhere),
    #      but the main requirement is that it be vertically centered.
    #
    #    We'll center both horizontally & vertically:
    canvas_cx = big_side / 2.0
    canvas_cy = big_side / 2.0

    # Final translation
    tx = canvas_cx - pivot_rot_x
    ty = canvas_cy - pivot_rot_y

    M_translate = np.array([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1]
    ], dtype=np.float32)

    # Combined transform
    M_final = M_translate @ M_rot  # 3x3
    M_affine = M_final[:2, :]

    # 7) Warp the image onto the big square
    output_image = cv2.warpAffine(
        image,
        M_affine,
        (big_side, big_side),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )

    return output_image
