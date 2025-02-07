import os
#
#sys.path.append(os.path.abspath(work_dir))
#append current dir to path
os.sys.path.append(os.getcwd())
from utils.common_tools import assemble_frames_to_video
from src.generation_routine import generate_sequence


work_dir = os.getcwd()
image_list = ["horizon_2",]#,"horizon_4","horizon_6","horizon_8","horizon_10","horizon_12"]

for i in image_list:
    base_image_name = i
    
    #set values
    this_image_path = work_dir+"/source_images_for_generation/base_images/"+base_image_name+".png"
    this_output_dir_dot="synth_videos/"+base_image_name+"/synth_track"
    this_output_dir_box="synth_videos/"+base_image_name+"/synth_track_box"
    this_output_dir_pose="synth_videos/"+base_image_name+"/synth_track_pose"
    this_log_file_name="synth_videos/"+base_image_name+"/frame_data.txt"
    this_zoom_start = 1
    this_zoom_end = 0.01 #0.5 is 2x zoom
    this_pitch_start_deg = 0
    this_pitch_end_deg = -30 #negative is up
    this_yaw_amplitude_deg = 1
    this_yaw_period=30 # frames per full left-right-left cycle
    this_dot_y_center=290
    this_dot_initial_x=1225
    this_dot_inter_frame_speed=1   # jump 10 px in global frame
    this_dot_size=1
    this_random_seed = 42
    #this_target_color = (114,114,181)
    #make a bright red color
    this_target_color = (0,0,255)
    synth_noise_level = 0.0001

    b = 1
    if b:
        generate_sequence(
        large_image_path= this_image_path,
        output_dir_dot=this_output_dir_dot,
        output_dir_box=this_output_dir_box,
        output_dir_pose=this_output_dir_pose,
        log_file_name = this_log_file_name,
        num_frames=60,
        output_width=1280,
        output_height=720,
        zoom_start=this_zoom_start,
        zoom_end=this_zoom_end,
        pitch_start_deg=this_pitch_start_deg,
        pitch_end_deg=this_pitch_end_deg,
        yaw_amplitude_deg=this_yaw_amplitude_deg,
        yaw_period=this_yaw_period,  
        dot_y_center=this_dot_y_center,
        dot_initial_x=this_dot_initial_x,
        dot_inter_frame_speed=this_dot_inter_frame_speed,   
        dot_size=this_dot_size,
        seed=this_random_seed,
        target_color =  this_target_color,
        synth_noise_level = synth_noise_level
        )

        videos_to_assemble = [this_output_dir_dot,this_output_dir_box,this_output_dir_pose]
    
        for v in videos_to_assemble:
            input_directory = v
            output_video = input_directory+"_video.mp4"
            frames_per_second = 15  # FPS for the video
            frame_glob_pattern = 'frame*.jpg'  # Pattern to match frames with bounding boxes
        
            try:
                assemble_frames_to_video(
                    input_dir=input_directory,
                    output_video_path=output_video,
                    fps=frames_per_second,
                    frame_pattern=frame_glob_pattern)
            except ValueError as ve:
                print(f"Error: {ve}")