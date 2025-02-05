import cv2, os, math, glob
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
from skimage import feature
from sklearn.linear_model import LinearRegression
from skimage.transform import rotate,warp


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.detection_tools import extract_object_and_background_masks
from utils.detection_tools import detect_sky, estimate_horizon_line_by_edges
from utils.detection_tools import rectify_horizon, downsampler, rotate_and_center_horizon
from utils.common_tools import annotate_image, show_bgr
from utils.common_tools import find_nonzero_bounding_box, trim_video, draw_parallel_lines

from utils.detection_tools import get_min_max_hsv, extract_contour_region
from auto_startup.config import ImageProcessingParams



def detect_basic(raw_frame,frame_number=None,debug=False,
                   ip_params = None,
    debug_image_width = 14):

    # if pass a frame with no context, assume there is 1 easy to detect object in the frame, and write to yaml

    ultra_raw_frame = raw_frame.copy()

    if ip_params is None:
        print("Warning: No ImageProcessingParams object was passed. Using default values.")
        ip_params = ImageProcessingParams(None)
    

    hf_noise_gaussian_kernel = ip_params.hf_noise_gaussian_kernel
    hf_noise_gaussian_sigma  = ip_params.hf_noise_gaussian_sigma
    sky_gaussian_kernel      = ip_params.sky_gaussian_kernel
    sky_gaussian_sigma       = ip_params.sky_gaussian_sigma
    hsv_lower_bound          = ip_params.hsv_lower_bound
    hsv_upper_bound          = ip_params.hsv_upper_bound
    adaptive_threshold_max_value = ip_params.adaptive_threshold_max_value
    adaptive_threshold_blockSize = ip_params.adaptive_threshold_blockSize
    adaptive_threshold_constant = ip_params.adaptive_threshold_constant
    sobel_pre_gaussian_kernel = ip_params.sobel_pre_gaussian_kernel
    sobel_pre_gaussian_sigma  = ip_params.sobel_pre_gaussian_sigma
    sobel_x_kernel = ip_params.sobel_x_kernel
    sobel_y_kernel = ip_params.sobel_y_kernel
    sobel_threshold = ip_params.sobel_threshold
    object_area_threshold = ip_params.object_area_threshold

    canny_threshold1 = 50
    canny_threshold2 = 150
    if debug:
        show_bgr(raw_frame, title=f"Raw Frame {frame_number}",
                    w=debug_image_width)
        
    base_image = raw_frame.copy()

 
    input_image = base_image.copy()
    # Convert to grayscale if needed
    gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

    adaptive_thresh = cv2.adaptiveThreshold(
        gray,
        maxValue=adaptive_threshold_max_value,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=adaptive_threshold_blockSize,
        C=adaptive_threshold_constant
    )


    blurred_gray = cv2.GaussianBlur(adaptive_thresh, 
                                    sobel_pre_gaussian_kernel, 
                                    sobel_pre_gaussian_sigma)

    if debug:
        show_bgr(blurred_gray, title=f"Pre-Sobel Blurred Gray, Frame {frame_number}",w=14)

    # Sobel
    sobel_x = cv2.Sobel(blurred_gray, cv2.CV_64F, 1, 0, ksize=sobel_x_kernel)
    sobel_y = cv2.Sobel(blurred_gray, cv2.CV_64F, 0, 1, ksize=sobel_y_kernel)
    sobel_mag = cv2.magnitude(sobel_x, sobel_y)
    sobel_abs = cv2.convertScaleAbs(sobel_mag)

    if debug:
        show_bgr(sobel_abs, title=f"Sobel Abs, Frame {frame_number}",
                 w=debug_image_width)

    # Threshold
    _, canny_edges = cv2.threshold(sobel_abs, sobel_threshold, 
                                  255, cv2.THRESH_BINARY)


    if debug:
        show_bgr(canny_edges, title=f"Edges in Frame, Frame {frame_number}",
                 w=debug_image_width)


    filtered_contours = None
    identified_object = None

    contours, _ = cv2.findContours(canny_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Filter contours to exclude those with an area larger than 400 pixels
        filtered_contours = [contour for contour in contours if cv2.contourArea(contour) <= object_area_threshold]

    if filtered_contours:
        # Find the largest remaining contour
        identified_object = max(filtered_contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(identified_object)
        bounding_box = (x, y, w, h)
    
    if identified_object is not None:

        M = cv2.moments(identified_object)

        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centroid = (cx, cy)
        else:
            # Fallback: if area is zero (degenerate contour),
            # pick the center of the bounding box
            centroid = (x + w // 2, y + h // 2)
            cx = x
            cy = y
        
        # Draw bounding box and centroid on the original image
        cv2.rectangle(base_image, (x-2*w, y-2*h), (x+4*w, y+4*h), (0, 0, 255), 4)

        extracted, contour_mask = extract_contour_region(ultra_raw_frame, identified_object)
        #{"min_h": min_h, "max_h": max_h, "min_s": min_s, "max_s": max_s, "min_v": min_v, "max_v": max_v}
        hsv_values =  get_min_max_hsv(ultra_raw_frame, contour_mask)
        print("HSV Values")
        print(hsv_values)
        #print(hsv_values)
        # Draw the contour on the ultra raw frame
        cv2.drawContours(ultra_raw_frame, [identified_object], -1, (0, 255, 0), 1)
        #Zoom in on the object
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        # Apply the closing operation.
        closed = cv2.morphologyEx(canny_edges, cv2.MORPH_CLOSE, kernel)

        # Optionally, find contours on the closed image:
        contours, hierarchy = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw the closed contour on a copy of the original image for visualization
        output = cv2.cvtColor(canny_edges, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(output, contours, -1, (0, 255, 0), 2)

        if debug:
            show_bgr(output, title=f"Closed Contours, Frame {frame_number}",
                     w=debug_image_width)
        zoomed_in = ultra_raw_frame[y-2:y+h+2, x-2:x+w+2]

        if debug:
            show_bgr(zoomed_in, title=f"Zoomed In, Frame {frame_number}",
                     w=debug_image_width)
            
            show_bgr(extracted, title=f"Extracted Region, Frame {frame_number}",
                     w=debug_image_width)
            
    else:
        contour_mask = None
        cx,cy,w,h = 0,0,0,0
    
    if frame_number:
        text = f"Frame: {frame_number} | Centroid: ({cx}, {cy}),{w}x{h} pixels"
    else:
        text = f"Centroid: ({cx}, {cy}),{w}x{h} pixels"

    annotate_image(base_image,text,text_size=1.5)

    if debug:
        show_bgr(base_image, title=f"Detected Object, Frame {frame_number}",
                 w=debug_image_width)
                 

    return base_image, cx,cy,w,h, contour_mask

