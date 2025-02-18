import cv2,sys,os
import matplotlib.pyplot as plt
import cv2, os, math, glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


os.sys.path.append(os.getcwd())


from utils.common_tools import annotate_image, show_bgr
from utils.common_tools import stack_videos, assemble_frames_to_video
from utils.common_tools import find_nonzero_bounding_box, trim_video, draw_parallel_lines

from utils.detection_tools import detect_sky, estimate_horizon_line_by_edges
from utils.detection_tools import  downsampler, rotate_and_center_horizon
from utils.common_tools import annotate_image, show_bgr, draw_parallel_lines
from src.detect_basic import detect_basic
from src.analysis_routine import calculate_errors



def analyze_video(video, truth_df,improc_params):

    base_image_name = "horizon_2"

    # Initialize new columns in the truth dataframe
    truth_df["detected_x"] = None
    truth_df["detected_y"] = None
    truth_df["detected_w"] = None
    truth_df["detected_h"] = None


    # Get total number of frames in the video
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in range(num_frames):

        # Retrieve true x and y from truth_df for the current frame
        # (Assumes truth_df has a 'frame_number' column)
        row = truth_df.loc[truth_df['frame_number'] == i]
        if row.empty:
            continue  # Skip frame if not found in truth_df

        true_x = row.iloc[0]['local_dot_x_truth']
        true_y = row.iloc[0]['local_dot_y_truth']
        print(f"True x: {true_x}, True y: {true_y}")
        # For the first frame, initialize ROI boundaries and expected dimensions
        if i == 0:
            y_min = int(true_y - 20)
            y_max = int( true_y + 20)
            x_min = int( true_x - 20)
            x_max = int( true_x + 20)
            initial_w = 4
            initial_h = 4

        print(f"True x: {true_x}, True y: {true_y}")

        if i % 10 == 0:
            print(f"processing frame {i}")

        # Set the frame position and read the frame
        video.set(cv2.CAP_PROP_POS_FRAMES, i)  # Using i as the frame index
        ret, raw_frame = video.read()
        if not ret:
            print(f"Frame {i} could not be read, skipping.")
            continue

        # Call the detection function (assumed to be defined elsewhere)
        # Unpack results from detect_basic (the first returned value is unused)
        _, cx, cy, w, h, _, _ = detect_basic(
            raw_frame, i, debug=False,
            ip_params=improc_params, save_figs=True,
            debug_image_width=14,
            b_range=None,
            o_range=None,
            y_min=y_min,
            y_max=y_max,
            x_min=x_min,
            x_max=x_max,
            expected_w=initial_w,
            expected_h=initial_h
        )

        print(f"Detected: cx={cx}, cy={cy}, w={w}, h={h}")

        offset = int(max(5*initial_w,5*initial_h))#max(40,int(8*initial_w))
        # Update ROI boundaries and expected dimensions for next frame
        if cy > 1 and cx > 1:
            y_min = cy - offset
            y_max = cy + offset
            x_min = cx - offset
            x_max = cx + offset
            initial_w = w*1.2
            initial_h = h*1.2
        else:
            print(f"Detection failed for frame {i}, skipping.")
            

        # Update truth_df with the detected values for this frame
        truth_df.loc[truth_df['frame_number'] == i, 'detected_x'] = cx
        truth_df.loc[truth_df['frame_number'] == i, 'detected_y'] = cy
        truth_df.loc[truth_df['frame_number'] == i, 'detected_w'] = w
        truth_df.loc[truth_df['frame_number'] == i, 'detected_h'] = h

    return truth_df
