import cv2, os, math, glob, sys
import numpy as np
import matplotlib.pyplot as plt
from skimage import feature
from sklearn.linear_model import LinearRegression
from skimage.transform import rotate,warp

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from utils.common_tools import annotate_image,show_bgr
from utils.generation_tools import add_random_noise, add_dot_and_bounding_box


def generate_sequence(
    large_image_path='images/base_images/horizon_2.png',
    output_dir_dot="synth_track",
    output_dir_box="synth_track_box",
    output_dir_pose="synth_track_pose",
    log_file_name = "frame_data.txt",
    num_frames=60,
    output_width=1920,
    output_height=1080,
    # Zoom range from 1.0 (no zoom) up to 2.0 (2x zoom)
    zoom_start=1,
    zoom_end=0.3,
    # Pitch range: start at 0° (flat), end at -15° (looking up)
    pitch_start_deg=0.0,
    pitch_end_deg=-45.0,
    # Yaw range: oscillate between -45° and +45°
    yaw_amplitude_deg=30.0,
    yaw_period=60,  # frames per full left-right-left cycle
    # Dot motion parameters
    dot_y_center=300,
    dot_initial_x=800,
    dot_inter_frame_speed=10,   # jump 10 px in x each frame
    dot_size=2,
    seed=42, #presently unused
    target_color = (9, 9, 171) ,
    synth_noise_level = 0
):
    """
    Generate 60 frames of a 1920x1080 "camera" viewing a larger image, with:
      - Zoom in from 1.0 to 2.0
      - Pitch upward from 0 to -15 deg
      - Yaw oscillation from -45 to +45 deg
    Also place a moving dot (and bounding box) in the large image before warping,
    so the dot warps in the final output.

    Outputs three sets of images:
      1) <output_dir_dot> : images with the dot only, warped to 1920x1080
      2) <output_dir_box> : images with the dot + bounding box, warped to 1920x1080
      3) <output_dir_pose>: large/base image with dot + bounding box + a drawn rectangle
                            showing the camera's FOV (i.e. src_corners).
    """
    if seed is not None:
        np.random.seed(seed) #presently unused

    # Read the large input image
    large_img_original = cv2.imread(large_image_path)
    if large_img_original is None:
        raise ValueError(f"Could not load '{large_image_path}'.")
    large_img_original = large_img_original[0:1400,0:large_img_original.shape[1]]
    H, W, _ = large_img_original.shape

    if W < output_width or H < output_height:
        raise ValueError("Input image must be larger than the requested output size.")

    # Make sure the output directories exist
    os.makedirs(output_dir_dot, exist_ok=True)
    os.makedirs(output_dir_box, exist_ok=True)
    os.makedirs(output_dir_pose, exist_ok=True)

    # Destination corners for the 1920x1080 output
    dst_corners = np.float32([
        [0,             0],
        [output_width-1, 0],
        [output_width-1, output_height-1],
        [0,             output_height-1]
    ])

    # We'll treat the center of the large image as the "look at" point
    cx, cy = W / 2.0, H / 2.0

    # Half-width/half-height of the camera in "local" coords
    half_w_out = output_width / 2.0
    half_h_out = output_height / 2.0

    #for data logging
    output_file = open(log_file_name, "w")
    output_file.write("frame_number,local_dot_x_truth,local_dot_y_truth,yaw_deg,pitch_deg,zoom,global_dot_x_truth,global_dot_y_truth,global_dot_size_truth\n")
    


    for i in range(num_frames):
        cy -=5
        frame_number = i + 1

        # Compute fraction t from 0..1 across frames
        t = i / float(num_frames - 1) if num_frames > 1 else 0

        # Current zoom
        zoom = zoom_start + t * (zoom_end - zoom_start)

        # Current pitch in degrees (negative => looking up)
        pitch_deg = pitch_start_deg + t * (pitch_end_deg - pitch_start_deg)
        pitch_rad = math.radians(pitch_deg)

        # Current yaw in degrees: oscillate +/- yaw_amplitude_deg
        yaw_deg = yaw_amplitude_deg * math.sin(2.0 * math.pi * i / yaw_period)
        yaw_rad = math.radians(yaw_deg)

        # ---- STEP 1: Place the moving dot in the large image before warping ----
        dot_x_center = dot_initial_x + dot_inter_frame_speed * i

        # Make a copy of the large image so we don't overwrite original
        large_img = large_img_original.copy()

        # Draw the dot and bounding box
        img_dot, img_box = add_dot_and_bounding_box(
            large_img,
            x_center=dot_x_center,
            method="smear",
            y_center=dot_y_center,
            object_size=dot_size,
            dot_color = target_color # BGR format
        )
        global_base_frame = img_dot.copy()
        # ---- STEP 2: Compute the source corners in "local" coords, then transform. ----
        # Base corners of the camera in local coords, centered at (0,0)
        local_corners = np.float32([
            [-half_w_out, -half_h_out],
            [ half_w_out, -half_h_out],
            [ half_w_out,  half_h_out],
            [-half_w_out,  half_h_out],
        ])

        # Apply zoom
        local_corners *= zoom

        # Apply in-plane yaw rotation
        cos_yaw = math.cos(yaw_rad)
        sin_yaw = math.sin(yaw_rad)
        R = np.array([
            [cos_yaw, -sin_yaw],
            [sin_yaw,  cos_yaw]
        ], dtype=np.float32)
        local_corners = local_corners @ R.T  # rotate around (0,0)

        # Approximate pitch by shifting top/bottom edges differently
        pitch_factor = 0.002
        y_range = half_h_out * zoom
        offset = pitch_rad * pitch_factor * y_range
        # Move top corners up by offset
        local_corners[0, 1] += offset
        local_corners[1, 1] += offset
        # Move bottom corners slightly the other way
        local_corners[2, 1] -= offset * 0.5
        local_corners[3, 1] -= offset * 0.5

        # Translate to the center of the large image
        src_corners = local_corners + np.float32([cx, cy])

        # Optional: clamp corners so they don't go out of bounds
        src_corners[:, 0] = np.clip(src_corners[:, 0], 0, W-1)
        src_corners[:, 1] = np.clip(src_corners[:, 1], 0, H-1)

        # ---- STEP 3: Warp the images (dot vs. box) ----
        M_dot = cv2.getPerspectiveTransform(src_corners, dst_corners)
        warped_dot = cv2.warpPerspective(img_dot, M_dot, (output_width, output_height))


        M_box = cv2.getPerspectiveTransform(src_corners, dst_corners)
        warped_box = cv2.warpPerspective(img_box, M_box, (output_width, output_height))

        
        # ---- STEP 4: Create a "pose" image showing camera FOV on the large image. ----
        # We'll overlay the rectangle representing the corners in 'src_corners'.
        # Let’s do that on top of the "img_box" version (the large image with dot+box),
        # or you can just as easily do it on top of 'img_dot' or the original large_img.
        img_pose = img_box.copy()  # full-size large image with dot and bounding box

        # Convert corners to integer coordinates
        corners_int = [(int(x), int(y)) for x, y in src_corners]

        # Draw lines between consecutive corners: (0->1->2->3->0)
        color_fov = (255, 0, 0)  # BGR (blue) or choose your favorite color
        thickness = 2
        for c_idx in range(4):
            c1 = corners_int[c_idx]
            c2 = corners_int[(c_idx + 1) % 4]
            cv2.line(img_pose, c1, c2, color_fov, thickness)

         # === STEP 5: Annotate the final warped_box image with the dot location in camera coords ===
        # We know (dot_x_center, dot_y_center) in the large image. 
        # Let's see where it lands after perspective transform M_box.
        src_dot_center = np.array([[[dot_x_center, dot_y_center]]], dtype=np.float32)
        dst_dot_center = cv2.perspectiveTransform(src_dot_center, M_box)
        dot_x_warped, dot_y_warped = dst_dot_center[0, 0]  # (x, y)

        # Draw a small circle in the warped image to show the dot center
        #cv2.circle(warped_box, (int(dot_x_warped), int(dot_y_warped)), 5, (0,255,0), -1)
        # Put text for debugging
        cv2.putText(
            warped_box,
            f"Dot=({dot_x_warped:.1f}, {dot_y_warped:.1f})",
            (int(dot_x_warped) + 10, int(dot_y_warped)),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (0,255,0),
            2
        )
        
        # ---- STEP 5: Save all three versions ----
        filename_dot = os.path.join(output_dir_dot, f"frame_{frame_number:03}.jpg")
        filename_box = os.path.join(output_dir_box, f"frame_{frame_number:03}_box.jpg")
        filename_pose = os.path.join(output_dir_pose, f"frame_{frame_number:03}_pose.jpg")

        text = f"Object: ({int(dot_x_center)}, {int(dot_y_center)}), {int(dot_size)} pixels wide | Camera: {int(yaw_deg)} Yaw, {int(pitch_deg)} Pitch, {zoom:.1f} Zoom"

        img_pose = annotate_image(img_pose,text)
        

        
        warped_dot = add_random_noise(warped_dot, noise_level=synth_noise_level)
        
        output_file.write(f"{i},{dot_x_warped},{dot_y_warped},{yaw_deg},{pitch_deg},{zoom},{dot_x_center},{dot_y_center},{dot_size}\n")

        dot_y_warped = int(dot_y_warped)
        dot_x_warped = int(dot_x_warped)
        #show_bgr(warped_dot)
        s1 = warped_dot[dot_y_warped-10:dot_y_warped+10,dot_x_warped-10:dot_x_warped+10]
        #show_bgr(s1)
        sy = (dot_y_center-dot_size*3)
        sx = (dot_x_center-dot_size*3)
        s_global = global_base_frame[sy:dot_y_center+dot_size*3,sx:dot_x_center+dot_size*3]
        #show_bgr(s_global)
        
        cv2.imwrite(filename_dot, warped_dot)
        cv2.imwrite(filename_box, warped_box)
        cv2.imwrite(filename_pose, img_pose)

        if i % 10 == 0:
            print(f"Saved:\n {filename_pose}")
            
    output_file.close()