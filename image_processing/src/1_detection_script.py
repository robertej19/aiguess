import cv2,sys,os
import matplotlib.pyplot as plt
import cv2, os, math, glob
import matplotlib.pyplot as plt
import numpy as np

os.sys.path.append(os.getcwd())


from utils.common_tools import annotate_image, show_bgr
from utils.common_tools import stack_videos, assemble_frames_to_video
from utils.common_tools import find_nonzero_bounding_box, trim_video, draw_parallel_lines

from utils.detection_tools import detect_sky, estimate_horizon_line_by_edges
from utils.detection_tools import  downsampler, rotate_and_center_horizon
from utils.common_tools import annotate_image, show_bgr, draw_parallel_lines
from src.detect_basic import detect_basic
from src.analysis_routine import calculate_errors


# create an image processing parameters class
class ImageProcessingParams:
    def __init__(self, adaptive_threshold_max_value=255, 
                 adaptive_threshold_blockSize=5, 
                 adaptive_threshold_constant=4,
                 sobel_pre_gaussian_kernel=[3,3],
                 sobel_pre_gaussian_sigma=0.5,
                 sobel_x_kernel=3,
                 sobel_y_kernel=3,
                 sobel_threshold=50,
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

def process_videos(base_name_list):
    for image_name in base_name_list:
        base_image_name = image_name

    # Paths for input and output
    truth_log_file_name = f"synth_videos/{base_image_name}/frame_data.txt"
    videos_to_process = [f"synth_videos/{base_image_name}/synth_track"]

    # Create the output directory for processed frames
    processed_frames_dir = f"synth_videos/{base_image_name}/processed_video_frames"
    rect_frames_dir = f"synth_videos/{base_image_name}/rectified_processed_video_frames"
    os.makedirs(processed_frames_dir, exist_ok=True)
    os.makedirs(rect_frames_dir, exist_ok=True)

    # Create or open the processed_data.txt file
    processed_data_file = open(f"synth_videos/{base_image_name}/processed_data.txt", "w")

    # Write the header for the processed_data.txt file
    processed_data_file.write("frame_index,cx,cy,w,h\n")

    for v in videos_to_process:
        video_to_process_path = v + "_video.mp4"
        cap = cv2.VideoCapture(video_to_process_path)

        if not cap.isOpened():
            raise ValueError(f"Error opening video file: {video_to_process_path}")




        background_lab_mean = [np.array([192 ,123,  91]),np.array( [202, 133 , 98])]
        object_lab_mean = [np.array([  140, 128, 121]), np.array([167, 190, 164])]
        #background_lab_mean = None
        #object_lab_mean = None
        for i in range(50, 55):
            if i % 10 == 0:
                print(f"processing frame {i} in {image_name}")
            frame_number = i
            # Set the frame position (0-based index)
            frame_index = frame_number - 1
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

            # Read the frame
            ret, raw_frame = cap.read()
            if not ret:
                print(f"Frame {frame_index} could not be read.")
                continue

            #raw_frame = raw_frame[0:720,0:1280]
            # Process the frame and get the output data          (frame_to_process,frame_number=None,debug=False,

                    
            output_frame, cx,cy,w,h, contour_mask, identified_object = detect_basic(raw_frame,i,debug=True,
                                                                                    ip_params = None,save_figs=True,
                                                                                    debug_image_width = 14,
                                                                                    b_range = background_lab_mean,
                                                                                    o_range = object_lab_mean)
            rectified_frame = output_frame.copy() #don't return recification anymore
            # Save the processed frame to the output directory
            output_frame_path = os.path.join(processed_frames_dir, f"frame_{frame_index:03}.jpg")
            rectified_frame_path = os.path.join(rect_frames_dir, f"frame_rect_{frame_index:03}.jpg")

            cv2.imwrite(output_frame_path, output_frame)
            cv2.imwrite(rectified_frame_path, rectified_frame)


            # Write the data for the current frame to processed_data.txt
            processed_data_file.write(f"{frame_index},{cx},{cy},{w},{h}\n")

    # Close the video capture and processed_data.txt file
    cap.release()
    processed_data_file.close()
    """

    output_video_path = f"synth_videos/{base_image_name}/processed_track_video.mp4" 
    output_rect_video_path = f"synth_videos/{base_image_name}/rectified_processed_track_video.mp4" 

    assemble_frames_to_video(processed_frames_dir, output_video_path, fps=5, frame_pattern='frame_*.jpg')
    print("Processed video assembled")
    assemble_frames_to_video(rect_frames_dir, output_rect_video_path, fps=5, frame_pattern='frame_*.jpg')
    print("Rectified processed video assembled")

    output_rectrim_video_path = f"synth_videos/{base_image_name}/processed_rect_trimmed_track_video.mp4"
    trim_video(output_rect_video_path , output_rectrim_video_path)
    print("Rectified processed video trimmed")

    stack_videos(video1_path = output_video_path,
                 video2_path = f"synth_videos/{base_image_name}/synth_track_box_video.mp4",
                 output_video_path = f"synth_videos/{base_image_name}/combined_video.mp4")
    print("Combined video assembled")
    stack_videos(video1_path = output_rect_video_path,
                 video2_path = f"synth_videos/{base_image_name}/synth_track_box_video.mp4",
                 output_video_path = f"synth_videos/{base_image_name}/rect_combined_video.mp4")
    print("Rectified combined video assembled")
    """

if __name__ == "__main__":
    base_name_list = ["horizon_2"]
    process_videos(base_name_list)
    print("Done!")
    truth_df_path = 'synth_videos/horizon_2/frame_data.txt'  # Columns: frame_number, local_dot_x_truth, local_dot_y_truth
    estimate_df = 'synth_videos/horizon_2/processed_data.txt' # Columns: frame_index, cx, cy
    calculate_errors(truth_df_path, estimate_df)