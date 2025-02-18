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
from src.vidana import analyze_video

# create an image processing parameters class
class ImageProcessingParams:
    def __init__(self, adaptive_threshold_max_value=255, 
                 adaptive_threshold_blockSize=7, 
                 adaptive_threshold_constant=5,
                 sobel_pre_gaussian_kernel=[5,5],
                 sobel_pre_gaussian_sigma=1,
                 sobel_x_kernel=3,
                 sobel_y_kernel=3,
                 sobel_threshold=20,
                 lab_offset=10,
                 object_w_max_threshold=20,
                 object_h_max_threshold=20):
        
        self.adaptive_threshold_max_value = adaptive_threshold_max_value
        self.adaptive_threshold_blockSize = adaptive_threshold_blockSize
        self.adaptive_threshold_constant = adaptive_threshold_constant
        self.sobel_pre_gaussian_kernel = sobel_pre_gaussian_kernel
        self.sobel_pre_gaussian_sigma = sobel_pre_gaussian_sigma
        self.sobel_x_kernel = sobel_x_kernel
        self.sobel_y_kernel = sobel_y_kernel
        self.sobel_threshold = sobel_threshold
        self.lab_offset = lab_offset
        self.object_w_max_threshold = object_w_max_threshold
        self.object_h_max_threshold = object_h_max_threshold



    

if __name__ == "__main__":

    improc_params = ImageProcessingParams()
    base_image_name = "horizon_2"
    videos_to_proces = f"synth_videos/{base_image_name}/synth_track"
    video_to_process_path = videos_to_proces + "_video.mp4"
    cap = cv2.VideoCapture(video_to_process_path)

    # Paths for input and output
    truth_log_file_name = f"synth_videos/{base_image_name}/frame_data.txt"
    
    #load the truth log file as a pandas dataframe
    truth_df = pd.read_csv(truth_log_file_name)

    truth_df_with_detections = analyze_video(cap, truth_df,improc_params)
    print(truth_df_with_detections)#.head())
    cap.release()
    
    # 3. Rename columns for clarity
    truth_df_with_detections.rename(
        columns={
            'local_dot_x_truth': 'truth_cx',
            'local_dot_y_truth': 'truth_cy',
            'detected_x': 'estimated_cx',
            'detected_y': 'estimated_cy'
        },
        inplace=True
    )

    # If you want to keep only the required columns:
    merged_df = truth_df_with_detections[['frame_number', 'truth_cx', 'truth_cy', 'estimated_cx', 'estimated_cy','detected_w','detected_h']]


    merged_df = merged_df.copy()
    merged_df['truth_cx'] = pd.to_numeric(merged_df['truth_cx'], errors='coerce')
    merged_df['estimated_cx'] = pd.to_numeric(merged_df['estimated_cx'], errors='coerce')
    merged_df['truth_cy'] = pd.to_numeric(merged_df['truth_cy'], errors='coerce')
    merged_df['estimated_cy'] = pd.to_numeric(merged_df['estimated_cy'], errors='coerce')


    # 4. Calculate Euclidean distance
    merged_df['distance'] = np.sqrt(
        (merged_df['truth_cx'] - merged_df['estimated_cx'])**2 +
        (merged_df['truth_cy'] - merged_df['estimated_cy'])**2
    )

    import numpy as np

    # Suppose 'merged_df' already has a 'distance' column computed
    # and a 'frame_number' column that starts at 0 for the first frame.
    total_frames = merged_df['frame_number'].nunique()  # or use max() + 1 if frames are contiguous

    # Create the 'penalty' column:
    merged_df['penalty'] = ((merged_df['frame_number'] + 1) / total_frames) * merged_df['distance']

    print(merged_df)

    # Print the sum of the penalty column:
    print("Total penalty:", merged_df['penalty'].sum())


    
    # 5. Plot frame_index on x-axis vs. distance on y-axis
    plt.figure(figsize=(10, 6))
    plt.plot(merged_df['frame_number'], merged_df['distance'], marker='o')
    plt.title('Distance between Truth and Estimated Points')
    plt.xlabel('Frame Index')
    plt.ylabel('Euclidean Distance')
    plt.grid(True)
    #plt.show()
    # save figure
    plt.savefig("Analysis_of_detrac.png")
    plt.close()
    # 6. (Optional) Inspect the merged data
    