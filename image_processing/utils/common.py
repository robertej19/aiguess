import cv2, os, math, glob
import numpy as np
import matplotlib.pyplot as plt
from skimage import feature
from sklearn.linear_model import LinearRegression
from skimage.transform import rotate,warp


def annotate_image(frame, cx, cy, w, h):
    # Prepare the text to be displayed
    text = f"Detected Centroid: ({cx}, {cy}), Size: {w}x{h} pixels"
    
    # Set the font, scale, color, and thickness for the text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2
    color = (0, 255, 255)  # Bright yellow color (BGR)
    thickness = 5
    
    # Get the text size to position it correctly at the bottom
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_width, text_height = text_size
    
    # Calculate the position for the text at the bottom of the image
    x_pos = (frame.shape[1] - text_width) // 2  # Centered horizontally
    y_pos = frame.shape[0] - 10  # Just above the bottom border
    
    # Add the text to the image
    cv2.putText(frame, text, (x_pos, y_pos), font, font_scale, color, thickness)
    
    return frame

def show_bgr(frame,w=20):
    plt.figure(figsize=(w,int(w/1.6)))
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    plt.imshow(frame_rgb)
    plt.show()

def annotate_truth(frame,yaw_deg,pitch_deg, zoom_val, cx,
                                    cy,
                                    object_size):
    # Prepare the text to be displayed
    text = f"Object: ({int(cx)}, {int(cy)}), {int(object_size)} pixels wide | Camera: {int(yaw_deg)} Yaw, {int(pitch_deg)} Pitch, {zoom_val:.1f} Zoom"

    # Set the font, scale, color, and thickness for the text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2
    color = (0, 255, 255)  # Bright yellow color (BGR)
    thickness = 5
    
    # Get the text size to position it correctly at the bottom
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_width, text_height = text_size
    
    # Calculate the position for the text at the bottom of the image
    x_pos = (frame.shape[1] - text_width) // 2  # Centered horizontally
    y_pos = frame.shape[0] - 10  # Just above the bottom border
    
    # Add the text to the image
    cv2.putText(frame, text, (x_pos, y_pos), font, font_scale, color, thickness)
    
    return frame

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

