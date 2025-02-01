
import cv2, os, math, glob
import numpy as np
import matplotlib.pyplot as plt
from skimage import feature
from sklearn.linear_model import LinearRegression
from skimage.transform import rotate,warp

def add_random_noise(frame, noise_level=0.1):
    """
    Add random noise to a frame.
    
    Parameters:
        frame (numpy.ndarray): The input image/frame.
        noise_level (float): The noise intensity, range [0, 1]. 
                             Higher values add more noise.

    Returns:
        numpy.ndarray: The noisy frame.
    """
    # Ensure noise level is between 0 and 1
    noise_level = np.clip(noise_level, 0, 1)

    # Generate random noise
    noise = np.random.randn(*frame.shape) * 255  # Noise values in range [-255, 255]

    # Scale the noise by the noise level
    noise = noise * noise_level

    # Add the noise to the original frame
    noisy_frame = frame + noise

    # Clip the values to ensure they remain valid pixel values [0, 255]
    noisy_frame = np.clip(noisy_frame, 0, 255).astype(np.uint8)

    return noisy_frame

def add_dot_and_bounding_box(
    image,
    object_size=10,
    method="smear",                 # "random", "precise", or "precise_smear"
    x_center=None,
    y_center=None,
    xmin=None,
    xmax=None,
    ymin=None,
    ymax=None,
    dot_color=None
):
    """
    Combines functionality of:
        1) add_dot_and_bounding_box (random center within [xmin, xmax, ymin, ymax])
        2) add_dot_and_bounding_box_precisely (exact center [x_center, y_center], no smear)
        3) add_dot_and_bounding_box_precisely_with_smear (exact center [x_center, y_center], with Gaussian smear)

    Parameters
    ----------
    image : np.ndarray
        Input image.
    object_size : int
        Radius of the dot (and determines bounding box size).
    method : str
        One of {"random", "precise", "precise_smear"}.
    x_center : int, optional
        X-coordinate of the dot center (for 'precise' or 'precise_smear').
    y_center : int, optional
        Y-coordinate of the dot center (for 'precise' or 'precise_smear').
    xmin, xmax, ymin, ymax : int, optional
        Bounding box limits for random center (used for 'random').
    dot_color : tuple(B, G, R), optional
        Color of the dot. If None, defaults are chosen based on `method`.

    Returns
    -------
    image_with_dot : np.ndarray
        Image where the dot has been placed.
    image_with_box : np.ndarray
        Image where the dot has been placed and a red bounding box is drawn around it.
    """
    # Copy the image to ensure we don't modify the input
    image_with_dot = image.copy()

    # Infer default dot color if none provided
    if dot_color is None:
        if method == "random":
            # Dark Orange-ish or the (0, 1, 1) used in original code
            dot_color = (0, 1, 1)
        else:
            # A grayish color used in the "precise" examples
            dot_color = (94, 114, 117)

    height, width = image.shape[:2]

    # -------------------------------------------------
    # 1) Compute the dot location
    # -------------------------------------------------
    if method == "random":
        # We need xmin, xmax, ymin, ymax
        if None in (xmin, xmax, ymin, ymax):
            raise ValueError(
                "For method='random', you must provide xmin, xmax, ymin, ymax."
            )

        # Ensure bounding box can contain the dot
        if (xmax - xmin) < 2 * object_size or (ymax - ymin) < 2 * object_size:
            raise ValueError("Bounding box is too small to contain the dot.")

        # Pick a random center
        center_x = np.random.randint(xmin + object_size, xmax - object_size)
        center_y = np.random.randint(ymin + object_size, ymax - object_size)

    elif method in ("precise", "smear"):
        # We need x_center, y_center
        if x_center is None or y_center is None:
            raise ValueError(
                f"For method='{method}', you must provide x_center and y_center."
            )

        # Validate that the dot will be fully within the image boundaries
        if not (object_size <= x_center < width - object_size and
                object_size <= y_center < height - object_size):
            raise ValueError(
                "The dot (with the specified object_size) does not fit inside the image boundaries."
            )

        center_x, center_y = x_center, y_center

    else:
        raise ValueError("method must be one of {'random', 'precise', 'smear'}")

    # -------------------------------------------------
    # 2) Draw the dot
    # -------------------------------------------------
    if method == "smear":
        # Create a Gaussian mask
        y_coords, x_coords = np.ogrid[:height, :width]
        distance = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)

        sigma = object_size / 2.0
        gaussian_mask = np.exp(-0.5 * (distance / sigma)**2)
        gaussian_mask = np.clip(gaussian_mask, 0, 1)

        # Blend dot_color into the image
        for c in range(3):
            image_with_dot[..., c] = np.clip(
                image_with_dot[..., c] * (1 - gaussian_mask) + 
                dot_color[c] * gaussian_mask,
                0, 255
            )
    else:
        # Simply draw a filled circle
        cv2.circle(image_with_dot, (center_x, center_y), object_size, dot_color, -1)

    # -------------------------------------------------
    # 3) Draw bounding box on a copy
    # -------------------------------------------------
    image_with_box = image_with_dot.copy()

    # Use bright red for the bounding box
    box_color = (0, 0, 255)  # BGR
    if method == "random":
        # The original random code used a bounding box size = max(50, object_size*2)
        box_size = max(50, object_size * 2)
        box_thickness = max(5, object_size // 5)
    else:
        # The "precise" methods used box_size = object_size * 2
        box_size = object_size * 2
        box_thickness = max(2, object_size // 5)

    top_left = (center_x - box_size, center_y - box_size)
    bottom_right = (center_x + box_size, center_y + box_size)

    # Ensure bounding box is within image boundaries
    top_left = (max(top_left[0], 0), max(top_left[1], 0))
    bottom_right = (
        min(bottom_right[0], width - 1),
        min(bottom_right[1], height - 1)
    )

    cv2.rectangle(image_with_box, top_left, bottom_right, box_color, box_thickness)

    return image_with_dot, image_with_box


##### DOT WITH SMEAR
def add_dot_and_bounding_box_precisely_with_smear(image, 
                                                  x_center, 
                                                  y_center, 
                                                  object_size,
                                                  dot_color=(94,114,117)):
    # Create copies of the original image for both outputs
    image_with_dot = image.copy()

    # Get image dimensions
    height, width = image.shape[:2]

    # Validate that the dot will be fully within the image boundaries
    if not (object_size <= x_center < width - object_size and
            object_size <= y_center < height - object_size):
        raise ValueError("The dot with the specified object_size does not fit within the image boundaries at the given center coordinates.")

    center = (x_center, y_center)


    
    # Create a Gaussian mask for the dot's fading effect
    y, x = np.ogrid[:image.shape[0], :image.shape[1]]
    distance_from_center = np.sqrt((x - x_center)**2 + (y - y_center)**2)
    
    # Generate a Gaussian distribution centered at (x_center, y_center)
    sigma = object_size / 2.0  # Set the standard deviation based on object size
    gaussian_mask = np.exp(-0.5 * (distance_from_center / sigma) ** 2)
    
    # Normalize the mask to have values between 0 and 1
    gaussian_mask = np.clip(gaussian_mask, 0, 1)

    # Now, we use the Gaussian mask to apply the color with fading effect
    for c in range(3):  # Apply the effect for each channel (BGR)
        image_with_dot[..., c] = np.clip(image_with_dot[..., c] * (1 - gaussian_mask) + dot_color[c] * gaussian_mask, 0, 255)

    # Define the color for the bounding box (Bright Red in BGR)
    box_color = (0, 0, 255)  # BGR format
    box_thickness = max(2, object_size // 5)  # Adjust thickness based on object size

    # Calculate the top-left and bottom-right points of the bounding box
    box_size = object_size * 2
    top_left = (x_center - box_size, y_center - box_size)
    bottom_right = (x_center + box_size, y_center + box_size)

    # Ensure the bounding box is within image boundaries
    top_left = (max(top_left[0], 0), max(top_left[1], 0))
    bottom_right = (min(bottom_right[0], width - 1), min(bottom_right[1], height - 1))

    image_with_box = image_with_dot.copy()

    # Draw the bounding box on the second image
    cv2.rectangle(image_with_box, top_left, bottom_right, box_color, box_thickness)

    return image_with_dot, image_with_box


def add_dot_and_bounding_box_precisely(image, x_center, y_center, object_size):
    # Create copies of the original image for both outputs
    image_with_dot = image.copy()

    # Get image dimensions
    height, width = image.shape[:2]
    

    # Validate that the dot will be fully within the image boundaries
    if not (object_size <= x_center < width - object_size and
            object_size <= y_center < height - object_size):
        raise ValueError("The dot with the specified object_size does not fit within the image boundaries at the given center coordinates.")

    center = (x_center, y_center)

    # Define the color for the dot (Dark Orange in BGR)
    #dot_color = (44,53,56)  # BGR format
    dot_color = (94,114,117)  # BGR format

    # Draw the filled circle (dot) directly on the image
    cv2.circle(image_with_dot, center, object_size, dot_color, -1)

    # Define the color for the bounding box (Bright Red in BGR)
    box_color = (0, 0, 255)  # BGR format
    box_thickness = max(2, object_size // 5)  # Adjust thickness based on object size

    # Calculate the top-left and bottom-right points of the bounding box
    box_size = object_size * 2
    top_left = (x_center - box_size, y_center - box_size)
    bottom_right = (x_center + box_size, y_center + box_size)

    # Ensure the bounding box is within image boundaries
    top_left = (max(top_left[0], 0), max(top_left[1], 0))
    bottom_right = (min(bottom_right[0], width - 1), min(bottom_right[1], height - 1))

    image_with_box = image_with_dot.copy()
    
    # Draw the bounding box on the second image
    cv2.rectangle(image_with_box, top_left, bottom_right, box_color, box_thickness)

    return image_with_dot, image_with_box


def generate_simple_video():
    # Video properties
    width, height = 1920, 1080
    fps = 30
    duration_seconds = 10
    num_frames = fps * duration_seconds

    # Define the video codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can use other codecs like 'XVID'
    video_filename = 'red_dot_video.mp4'
    out = cv2.VideoWriter(video_filename, fourcc, fps, (width, height))

    # Dot properties
    radius = 50  # Radius in pixels (10 pixels in diameter)
    color = (0, 0, 255)  # Red color in BGR format
    color2 = (0, 255,0)  # Red color in BGR format

    thickness = -1     # Filled circle

    # Starting position (center of the frame) and velocity (pixels per frame)
    x, y = width // 2, height // 2
    dx, dy = 5, 3  # Change these values for different speeds

    for frame_idx in range(num_frames):
        # Create a black background
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Update position
        x += dx
        y += dy
        
        # Bounce off the left/right walls
        if x - radius < 0 or x + radius > width:
            dx = -dx
            x += dx  # Adjust position after bouncing

        # Bounce off the top/bottom walls
        if y - radius < 0 or y + radius > height:
            dy = -dy
            y += dy  # Adjust position after bouncing

        # Draw the red dot on the frame
        #cv2.circle(frame, (x, y), radius, color, thickness)
        # Draw rectangle on the frame
        #add random noise to color
        color = (0,0,255)
        color = tuple([c + np.random.randint(-50,50) for c in color])
        cv2.rectangle(frame, (x, y), (x+radius, y+radius), color, thickness)

        #offset = 100
        #cv2.rectangle(frame, (x-offset, y-offset), (x-offset+radius, y-offset+radius), color2, thickness)

        
        # Write the frame to the video file
        out.write(frame)

    # Release the video writer and cleanup
    out.release()
    print(f"Video saved as {video_filename}")

# if name main run generate_simple_video
if __name__ == "__main__":
    generate_simple_video()