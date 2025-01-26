import cv2, os, math, glob
import numpy as np
import matplotlib.pyplot as plt
from skimage import feature
from sklearn.linear_model import LinearRegression
from skimage.transform import rotate,warp


def find_nonzero_bounding_box(frame):
    """
    Find the smallest bounding box that contains all non-zero pixels in the frame.

    Parameters
    ----------
    frame : np.ndarray
        Input frame.

    Returns
    -------
    tuple
        Bounding box coordinates (x_min, x_max, y_min, y_max).
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    non_zero_indices = cv2.findNonZero(gray)

    if non_zero_indices is None:
        return 0, frame.shape[1], 0, frame.shape[0]  # No non-zero pixels, keep full frame

    x_min, y_min, w, h = cv2.boundingRect(non_zero_indices)
    x_max = x_min + w
    y_max = y_min + h

    return x_min, x_max, y_min, y_max

def trim_video(input_path, output_path):
    """
    Trim an input video to remove zero padding from each frame.

    Parameters
    ----------
    input_path : str
        Path to the input video file.
    output_path : str
        Path to save the trimmed output video.
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {input_path}")

    # Initialize variables
    global_x_min, global_x_max = None, None
    global_y_min, global_y_max = None, None

    # Determine the bounding box across all frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        x_min, x_max, y_min, y_max = find_nonzero_bounding_box(frame)

        if global_x_min is None:
            global_x_min, global_x_max = x_min, x_max
            global_y_min, global_y_max = y_min, y_max
        else:
            global_x_min = min(global_x_min, x_min)
            global_x_max = max(global_x_max, x_max)
            global_y_min = min(global_y_min, y_min)
            global_y_max = max(global_y_max, y_max)

    cap.release()

    # Crop dimensions
    crop_width = global_x_max - global_x_min
    crop_height = global_y_max - global_y_min

    # Reopen video for trimming
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (crop_width, crop_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Crop frame using the determined global bounding box
        trimmed_frame = frame[global_y_min:global_y_max, global_x_min:global_x_max]

        # Write trimmed frame to output video
        out.write(trimmed_frame)

    cap.release()
    out.release()

# Example usage



def annotate_image(frame, text,x_pos=None,y_pos=None):
    # Prepare the text to be displayed
    
    # Set the font, scale, color, and thickness for the text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2
    color = (0, 255, 255)  # Bright yellow color (BGR)
    thickness = 5
    
    # Get the text size to position it correctly at the bottom
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_width, text_height = text_size
    
    # Calculate the position for the text at the bottom of the image
    if x_pos is None:
        x_pos = (frame.shape[1] - text_width) // 2  # Centered horizontally
    if y_pos is None:
        y_pos = frame.shape[0] - 10  # Just above the bottom border
    
    # Add the text to the image
    cv2.putText(frame, text, (x_pos, y_pos), font, font_scale, color, thickness)
    
    return frame

def show_bgr(frame,w=5,title=""):
    plt.figure(figsize=(w,int(w/1.6)))
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    plt.imshow(frame_rgb)
    plt.title(title)
    plt.show()


def assemble_frames_to_video(input_dir, output_video_path, fps=15, frame_pattern='frame_*.jpg'):
    """
    Assembles all image frames in the specified directory into a video.

    Parameters:
        input_dir (str): Path to the directory containing the image frames.
        output_video_path (str): Path where the output video will be saved (e.g., 'output_video.mp4').
        fps (int, optional): Frames per second for the output video. Default is 15.
        frame_pattern (str, optional): Glob pattern to match frame filenames. Default is 'frame_*.jpg'.

    Raises:
        ValueError: If no frames are found in the specified directory.
    """
    # Get list of frame file paths matching the pattern
    frame_files = sorted(glob.glob(os.path.join(input_dir, frame_pattern)))

    if not frame_files:
        raise ValueError(f"No frames found in directory '{input_dir}' with pattern '{frame_pattern}'.")

    # Read the first frame to get frame dimensions
    first_frame = cv2.imread(frame_files[0])
    if first_frame is None:
        raise ValueError(f"Unable to read the first frame: {frame_files[0]}")

    height, width, channels = first_frame.shape

    # Define the codec and create VideoWriter object
    # For MP4 output, use 'mp4v' codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    print(f"Starting video assembly from {len(frame_files)} frames...")

    for idx, frame_file in enumerate(frame_files):
        frame = cv2.imread(frame_file)
        if frame is None:
            print(f"Warning: Skipping unreadable frame: {frame_file}")
            continue

        # Check if frame size matches the first frame
        if frame.shape != first_frame.shape:
            print(f"Warning: Skipping frame with mismatched size: {frame_file}")
            continue

        video_writer.write(frame)

        if (idx + 1) % 10 == 0 or (idx + 1) == len(frame_files):
            print(f"Processed {idx + 1}/{len(frame_files)} frames.")

    # Release the VideoWriter
    video_writer.release()
    print(f"Video successfully saved to '{output_video_path}'.")

def stack_videos(video1_path,
                 video2_path,
                 output_video_path):
     
    # Open video files
    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)

    # Get properties of the first video
    fps1 = int(cap1.get(cv2.CAP_PROP_FPS))
    width1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Get properties of the second video
    fps2 = int(cap2.get(cv2.CAP_PROP_FPS))
    width2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
    height2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Ensure both videos have the same FPS (or choose one FPS as the standard)
    fps = min(fps1, fps2)

    # Determine the new width for resizing (use the smaller width)
    new_width = min(width1, width2)

    # Calculate aspect ratios and determine new heights
    new_height1 = int(height1 * (new_width / width1))
    new_height2 = int(height2 * (new_width / width2))

    # Output video dimensions
    output_height = new_height1 + new_height2
    output_width = new_width

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can use 'XVID' or other codecs
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (output_width, output_height))

    while True:
        # Read frames from both videos
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        # Stop if either video ends
        if not ret1 or not ret2:
            break

        # Resize frames to the new dimensions
        frame1_resized = cv2.resize(frame1, (new_width, new_height1))
        frame2_resized = cv2.resize(frame2, (new_width, new_height2))

        # Stack the frames vertically
        combined_frame = cv2.vconcat([frame1_resized, frame2_resized])

        # Write the combined frame to the output video
        out.write(combined_frame)

    # Release resources
    cap1.release()
    cap2.release()
    out.release()

    print(f"Combined video saved at {output_video_path}")



def is_line_within_mask(sky_mask, line_start, line_end):
    # Create a blank mask with the same size as the sky_mask
    line_mask = np.zeros_like(sky_mask, dtype=np.uint8)

    # Draw the line on the blank mask (white line)
    cv2.line(line_mask, line_start, line_end, 255, 1)

    # Check overlap between the line_mask and the sky_mask
    overlap = cv2.bitwise_and(sky_mask, line_mask)

    # If any overlap exists, the line is within the mask
    return np.any(overlap > 0)

def draw_parallel_lines(frame,sky_mask, slope, intercept, distance=10):
    # Height and width of the frame
    height, width, _ = frame.shape
    
    # Compute direction vector (dx, dy) for the line
    dx = 1
    dy = slope
    
    # Normalize the direction vector to unit length
    length = np.sqrt(dx ** 2 + dy ** 2)
    dx /= length
    dy /= length
    
    # Perpendicular vector (dy, -dx) scaled by distance
    perp_dx = -dy * distance
    perp_dy = dx * distance

    # Calculate original line endpoints
    x1, y1 = 0, intercept
    x2, y2 = width, slope * width + intercept
    
    # Calculate parallel line endpoints
    blue_x1, blue_y1 = x1 + perp_dx, y1 + perp_dy
    blue_x2, blue_y2 = x2 + perp_dx, y2 + perp_dy
    
    red_x1, red_y1 = x1 - perp_dx, y1 - perp_dy
    red_x2, red_y2 = x2 - perp_dx, y2 - perp_dy

    # Check if the blue or red line is within the sky_mask
    blue_within_sky = is_line_within_mask(sky_mask, (int(blue_x1), int(blue_y1)), (int(blue_x2), int(blue_y2)))
    red_within_sky = is_line_within_mask(sky_mask, (int(red_x1), int(red_y1)), (int(red_x2), int(red_y2)))

    # Display results
    print("Blue line within sky_mask:", blue_within_sky)
    print("Red line within sky_mask:", red_within_sky)




    # Determine the direction and line to use
    if blue_within_sky:
        selected_line_start = (int(blue_x1), int(blue_y1))
        selected_line_end = (int(blue_x2), int(blue_y2))
        perpendicular_direction = (int(-100 * dy), int(100 * dx))  # Direction from blue line
    elif red_within_sky:
        selected_line_start = (int(red_x1), int(red_y1))
        selected_line_end = (int(red_x2), int(red_y2))
        perpendicular_direction = (int(100 * dy), int(-100 * dx))  # Direction from red line
    else:
        selected_line_start = None
        selected_line_end = None

    # Create the region mask
    if selected_line_start and selected_line_end:
        region_mask = create_one_sided_region_mask(sky_mask, selected_line_start, selected_line_end, perpendicular_direction)
    else:
        print("No line is within the sky_mask.")


    # Draw the original line (optional for reference)
    cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 1)

    # Draw the blue parallel line
    cv2.line(frame, (int(blue_x1), int(blue_y1)), (int(blue_x2), int(blue_y2)), (255, 0, 0), 2)

    # Draw the red parallel line
    cv2.line(frame, (int(red_x1), int(red_y1)), (int(red_x2), int(red_y2)), (0, 0, 255), 2)

    return frame, region_mask


def create_one_sided_region_mask(sky_mask, line_start, line_end, perpendicular_direction):
    # Create a blank mask with the same size as the sky_mask
    region_mask = np.zeros_like(sky_mask, dtype=np.uint8)

    # Define the perpendicular direction (dx, dy)
    dx, dy = perpendicular_direction

    # Extend the line to form a region
    height, width = sky_mask.shape
    if dx > 0:
        # Extend to the right edge
        extended_start = (width, int(line_start[1] + (width - line_start[0]) * (dy / dx)))
        extended_end = (width, int(line_end[1] + (width - line_end[0]) * (dy / dx)))
    else:
        # Extend to the left edge
        extended_start = (0, int(line_start[1] - line_start[0] * (dy / dx)))
        extended_end = (0, int(line_end[1] - line_end[0] * (dy / dx)))

    # Define the polygon points for the region
    polygon_points = np.array([
        [line_start, line_end, extended_end, extended_start]
    ], dtype=np.int32)

    # Fill the polygon on the mask
    cv2.fillPoly(region_mask, polygon_points, 255)

    # Combine the region mask with the sky mask
    final_mask = cv2.bitwise_and(sky_mask, region_mask)

    return final_mask