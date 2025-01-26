import cv2, os, math, glob
import numpy as np
import matplotlib.pyplot as plt
from skimage import feature
from sklearn.linear_model import LinearRegression
from skimage.transform import rotate,warp


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