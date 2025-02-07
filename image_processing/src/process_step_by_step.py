import cv2

import os
import time
import shutil
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.detection_tools import get_min_max_lab_values, create_donut_mask_with_exclusion
from src.detect_basic import detect_basic
from utils.common_tools import numbered_framing_from_ascii, show_bgr

def main():


    output_filename = f"frame_of_interest_{video_frame_number}.png"
    video_frame_number = 30

    if len(sys.argv) < 2:
        print("Using {output_filename}")
    elif len(sys.argv) == 3:
        video_path= sys.argv[1]
        video_frame_number = int(sys.argv[2])
        print(f"Using frame number {video_frame_number} from {video_path}")
    else:
        print("Usage: python process_step_by_step.py <video_path> <frame_number>")
        sys.exit(1)



    output_filename = f"frame_of_interest_{video_frame_number}.png"
    output_path = os.path.join("resources/", output_filename)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Error: Could not open video at {video_path}")
    
    # Set the video to the desired frame number.
    cap.set(cv2.CAP_PROP_POS_FRAMES, video_frame_number)
    
    # Read the frame.
    ret, frame = cap.read()
    if not ret:
        cap.release()
        raise ValueError(f"Error: Could not read frame number {video_frame_number}")
    
    show_bgr(frame,w=5,title="",save_fig=True,save_fig_name=output_path)
    



if __name__ == "__main__":
    main()


 
