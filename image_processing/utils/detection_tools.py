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

def rotate_and_center_horizon(image, slope, intercept, upside_down=0):
    """
    Rotate 'image' so that the line y = slope*x + intercept becomes horizontal
    and is both horizontally and vertically centered in a fixed-size output canvas.
    If the frame is upside down, flip it so that the sky is at the top.

    The output canvas is a square of side = 2 * max(W, H).

    Parameters
    ----------
    image : np.ndarray
        Input image of shape (H, W).
    slope : float
        Slope (m) of horizon line in the original image (y = m*x + b).
    intercept : float
        Intercept (b) of the horizon line in the original image.
    upside_down : int
        If 1, flips the final image vertically to ensure the sky is at the top.

    Returns
    -------
    output_image : np.ndarray
        Rotated/translated image in a canvas of size:
          (2 * max(W, H), 2 * max(W, H)).
        The horizon line is horizontal and centered vertically and horizontally.
        If upside_down=1, the output is flipped vertically.
    """
    # 1) Dimensions of input
    H, W = image.shape[:2]
    big_side = 2 * max(W, H)  # final canvas dimension for both width and height

    # 2) Compute the rotation angle to flatten the horizon
    angle_radians = -np.arctan(slope)
    cos_a = np.cos(angle_radians)
    sin_a = np.sin(angle_radians)

    # 3) Choose pivot on the horizon at x = W/2
    pivot_x = W / 2.0
    pivot_y = slope * pivot_x + intercept

    # 4) Build 3x3 rotation-around-pivot matrix
    M_rot = np.array([
        [ cos_a, -sin_a,  pivot_x - pivot_x*cos_a + pivot_y*sin_a],
        [ sin_a,  cos_a,  pivot_y - pivot_x*sin_a - pivot_y*cos_a],
        [     0,      0,                                 1       ]
    ], dtype=np.float32)

    # 5) Find pivot location AFTER rotation
    pivot_original = np.array([pivot_x, pivot_y, 1.0], dtype=np.float32)
    pivot_rotated = M_rot @ pivot_original
    pivot_rot_x, pivot_rot_y, _ = pivot_rotated

    # 6) We want the pivot to be at the canvas center
    canvas_cx = big_side / 2.0
    canvas_cy = big_side / 2.0

    # Calculate translation offsets
    tx = canvas_cx - pivot_rot_x
    ty = canvas_cy - pivot_rot_y

    # Build translation matrix
    M_translate = np.array([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1 ]
    ], dtype=np.float32)

    # Combine the rotation + translation matrices
    M_final = M_translate @ M_rot
    M_affine = M_final[:2, :]  # Extract the 2x3 affine transformation

    # 7) Warp the image onto the big square
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




def remove_sky_band_morph(
    sky_mask, 
    slope, 
    intercept, 
    thickness=5, 
    band_width=10
):
    """
    Removes a band of sky near the horizon line using a morphological approach:
    1) Draw a line mask for the horizon.
    2) Dilate that line to 'band_width' so it covers a boundary region near the horizon.
    3) Subtract that band from sky_mask.

    Parameters
    ----------
    sky_mask : np.ndarray (H,W), dtype=uint8 or bool
        Binary mask (1=sky, 0=not sky).
    slope : float
        Slope m in y = m*x + b.
    intercept : float
        Intercept b in y = m*x + b.
    thickness : int
        Thickness in pixels to draw the initial horizon line.
    band_width : int
        How wide (in pixels) of a band around the horizon line to remove.

    Returns
    -------
    new_sky_mask : np.ndarray (H,W) 
        Sky mask with a morphological band near the horizon removed.
    """

    # Ensure mask is 8-bit
    sky_mask = sky_mask.astype(np.uint8)
    H, W = sky_mask.shape[:2]

    # 1) Draw horizon line in a separate mask
    horizon_mask = np.zeros((H, W), dtype=np.uint8)

    # We'll sample x from 0..W-1, compute y. If it's in [0..H-1], draw the line.
    # For a more accurate line, we can do cv2.line with two endpoints:
    #   x1=0, y1=int(b) and x2=W-1, y2=int(m*(W-1) + b).
    # But let's do a param approach:
    x1, y1 = 0, slope*0 + intercept
    x2, y2 = W-1, slope*(W-1) + intercept

    # Round and clamp
    y1 = int(round(np.clip(y1, 0, H-1)))
    y2 = int(round(np.clip(y2, 0, H-1)))

    # Now draw the line
    cv2.line(
        horizon_mask,
        (0, y1),
        (W-1, y2),
        color=1,
        thickness=thickness
    )

    # 2) Dilate that line to 'band_width'
    # Create a kernel for dilation
    kernel_size = (band_width, band_width)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    band_mask = cv2.dilate(horizon_mask, kernel, iterations=1)

    # 3) Limit band_mask to where sky is 1. This is the boundary in the sky region.
    band_sky = cv2.bitwise_and(band_mask, sky_mask)

    # 4) Subtract band from the sky_mask => remove that boundary region
    new_sky_mask = cv2.bitwise_and(sky_mask, cv2.bitwise_not(band_sky))

    return new_sky_mask


import numpy as np

def remove_sky_band_distance(
    sky_mask, 
    slope, 
    intercept, 
    dist_threshold, 
    reference_sky_pixel=None
):
    """
    Removes sky pixels whose perpendicular distance to the horizon line
    y = slope*x + intercept is < dist_threshold. This effectively shaves off
    a band near the horizon, regardless of orientation.

    Parameters
    ----------
    sky_mask : np.ndarray (H,W), dtype=uint8 or bool
        Binary mask where sky_mask[y, x] = 1 indicates sky.
    slope : float
        Slope m in y = m*x + b.
    intercept : float
        Intercept b in y = m*x + b.
    dist_threshold : float
        Minimum distance from horizon line (in pixels) to keep. Anything closer is removed.
    reference_sky_pixel : (x, y) tuple or None
        A known sky pixel to determine which side of the line is sky.
        If None, the function automatically finds one from sky_mask.

    Returns
    -------
    new_sky_mask : np.ndarray (H,W)
        A modified sky mask where the band near the horizon line has been removed.
    """

    H, W = sky_mask.shape[:2]
    sky_mask = sky_mask.astype(np.uint8)  # ensure 0/1

    # 1. If reference_sky_pixel not given, pick any pixel from sky_mask
    if reference_sky_pixel is None:
        coords = np.argwhere(sky_mask > 0)
        if len(coords) == 0:
            # No sky pixels, just return the original
            return sky_mask
        # Pick the first found sky pixel
        ref_y, ref_x = coords[0]
    else:
        ref_x, ref_y = reference_sky_pixel

    # 2. Compute sign for the reference sky pixel
    #    The line is y = m*x + b => rearranging => m*x - y + b = 0
    #    Signed distance sign: sign( m*ref_x - ref_y + b )
    ref_val = slope * ref_x - ref_y + intercept
    # If ref_val > 0 => sky is on the "positive side" of the line, else negative side
    sky_side = 1 if ref_val > 0 else -1

    # 3. Vectorized distance check
    #    We'll gather all sky pixels, compute their signed distance
    #    dist_signed = (m*x - y + b) / sqrt(m^2 + 1)
    denom = np.sqrt(slope**2 + 1.0)
    ys, xs = np.where(sky_mask == 1)
    dist_signed = (slope * xs - ys + intercept) / denom

    # Determine which pixels to keep
    # Condition A: sign(dist_signed) == sky_side
    # Condition B: abs(dist_signed) >= dist_threshold
    # So we keep those that satisfy both
    keep_mask = (np.sign(dist_signed) == sky_side) & (np.abs(dist_signed) >= dist_threshold)

    # 4. Build the new sky mask
    new_sky_mask = np.zeros_like(sky_mask, dtype=np.uint8)
    new_sky_mask[ys[keep_mask], xs[keep_mask]] = 1

    return new_sky_mask



