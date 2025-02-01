import cv2, os, math, glob
import numpy as np
import pandas as pd
import sys
import yaml

import matplotlib.pyplot as plt
from skimage import feature
from sklearn.linear_model import LinearRegression
from skimage.transform import rotate,warp

from utils.detection_tools import detect_sky, estimate_horizon_line_by_edges
from utils.detection_tools import rectify_horizon, downsampler, rotate_and_center_horizon
from utils.common_tools import annotate_image, show_bgr

from utils.common_tools import find_nonzero_bounding_box, trim_video, draw_parallel_lines

from utils.detection_tools import extract_object_and_background_masks
from utils.detection_tools import get_min_max_hsv, extract_contour_region


class ImageProcessingParams:
    def __init__(self, ip_config):
        self.hf_noise_gaussian_kernel = tuple(ip_config["hf_noise_gaussian_kernel"])
        self.hf_noise_gaussian_sigma  = ip_config["hf_noise_gaussian_sigma"]
        self.sky_gaussian_kernel      = tuple(ip_config["sky_gaussian_kernel"])
        self.sky_gaussian_sigma       = ip_config["sky_gaussian_sigma"]
        self.hsv_lower_bound         = tuple(ip_config["hsv_lower_bound"])
        self.hsv_upper_bound         = tuple(ip_config["hsv_upper_bound"])
        self.adaptive_threshold_max_value = ip_config["adaptive_threshold_max_value"]
        self.adaptive_threshold_blockSize  = ip_config["adaptive_threshold_block_size"]
        self.adaptive_threshold_constant   = ip_config["adaptive_threshold_constant"]
        self.sobel_pre_gaussian_kernel = tuple(ip_config["sobel_pre_gaussian_kernel"])
        self.sobel_pre_gaussian_sigma  = ip_config["sobel_pre_gaussian_sigma"]
        self.sobel_x_kernel = ip_config["sobel_x_kernel"]
        self.sobel_y_kernel = ip_config["sobel_y_kernel"]
        self.sobel_threshold = ip_config["sobel_threshold"]
        self.object_area_threshold = ip_config["object_area_threshold"]



def find_birds(raw_frame,frame_number=None,debug=False,
    debug_image_width = 14):#ip_params=None):
    ultra_raw_frame = raw_frame.copy()

    # 1) Load config
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    # 2) Create an ImageProcessingParams instance
    ip_params = ImageProcessingParams(config["image_processing"])

    """ip_params is an instance of ImageProcessingParams."""
    if ip_params is None:
        # fallback to defaults, or raise an error
        raise ValueError("ip_params is required for find_birds.")

    # Use ip_params just like normal attributes
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
    if debug:
        show_bgr(raw_frame, title=f"Raw Frame {frame_number}",
                    w=debug_image_width)
        
    base_image = raw_frame.copy()

 

    ## Detection is added by having an initial Gaussian blur
    ## Check that this is actually used
    base_image_small_blur = cv2.GaussianBlur(base_image,
                                             hf_noise_gaussian_kernel,hf_noise_gaussian_sigma)

    if debug:
        show_bgr(base_image_small_blur, 
                 title=f"Small Blurred, Frame {frame_number}",
                 w=debug_image_width)
        

    very_blurred_base_image = cv2.GaussianBlur(base_image, 
                                               sky_gaussian_kernel,
                                               sky_gaussian_sigma)
    
    if debug:
        show_bgr(very_blurred_base_image, 
                 title=f"Very Blurred, Frame {frame_number}",
                 w=debug_image_width)
        
    sky_mask = detect_sky(very_blurred_base_image)

    if debug:
        show_bgr(sky_mask, title=f"Sky Mask, Frame {frame_number}",
                 w=debug_image_width)
                 


    horizon_result = estimate_horizon_line_by_edges(sky_mask)


    #Need to handle edge case where slope = 1/0
    if horizon_result is None:
        # Fallback if horizon detection fails
        print("Warning: Horizon detection failed. Using default horizon.")
        slope, intercept = 0, sky_mask.shape[0]
        horizon_search_offset = 1
    else:
        slope, intercept = horizon_result
        horizon_search_offset = 50

    h,w = sky_mask.shape
    
    x_start, x_end = 0, w - 1
    y_start = int(slope * x_start + intercept)
    y_end   = int(slope * x_end   + intercept)
    #############################################
    ### SKY ROI CREATED
    ###########################################
    base_image_with_horizion = cv2.line(base_image, (x_start, y_start), (x_end, y_end), (0, 0, 255), 2)
    
    if debug:
        show_bgr(base_image_with_horizion, title=f"Horizon Line, Frame {frame_number}",
                 w=debug_image_width)

    base_image_x = base_image.copy()
    

    double_line, region_mask, upside_down =  draw_parallel_lines(base_image_x,sky_mask, slope, intercept, distance=horizon_search_offset)

    if debug:
        show_bgr(region_mask, title=f"Region Mask, Frame {frame_number}",
                 w=debug_image_width)
                 
        show_bgr(double_line, title=f"Double Line, Frame {frame_number}",
                 w=debug_image_width)
                 

    # Read an image (can be color or already grayscale)
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

    hsv_frame = raw_frame.copy()

    hsv_frame_hsv =cv2.cvtColor(hsv_frame, cv2.COLOR_BGR2HSV)

    lb = np.array(hsv_lower_bound, np.uint8)
    ub = np.array(hsv_upper_bound, np.uint8)
    hsv_mask = cv2.inRange(hsv_frame_hsv, lb, ub)

    if debug:
        show_bgr(hsv_mask, title=f"HSV Mask, Frame {frame_number}",w=14)
        show_bgr(adaptive_thresh, title=f"Adaptive Threshold, Frame {frame_number}",w=14)


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
    _, sobel_mask = cv2.threshold(sobel_abs, sobel_threshold, 
                                  255, cv2.THRESH_BINARY)

    if debug:
        show_bgr(sobel_mask, title=f"Sobel Mask, Frame {frame_number}",
                 w=debug_image_width)


    # Restrict edges to sky ROI
    edges_in_sky_roi = cv2.bitwise_and(sobel_mask, region_mask)
    edges_in_sky_roi_with_hsv_mask = cv2.bitwise_and(edges_in_sky_roi,hsv_mask)

        
    if debug:
        show_bgr(edges_in_sky_roi, title=f"Edges in Sky ROI, Frame {frame_number}",
                 w=debug_image_width)


    if debug:
        show_bgr(edges_in_sky_roi_with_hsv_mask, title=f"Edges in Sky ROI with HSV, Frame {frame_number}",
                 w=debug_image_width)



    filtered_contours = None
    identified_object = None

    contours, _ = cv2.findContours(edges_in_sky_roi_with_hsv_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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

        zoomed_in = ultra_raw_frame[y-2:y+h+2, x-2:x+w+2]

        if debug:
            show_bgr(zoomed_in, title=f"Zoomed In, Frame {frame_number}",
                     w=debug_image_width)
            
            show_bgr(extracted, title=f"Extracted Region, Frame {frame_number}",
                     w=debug_image_width)
            
    else:
        contour_mask = None
        cx,cy,w,h = 0,0,0,0
    
    rectified_frame = rotate_and_center_horizon(base_image,slope,intercept,upside_down=upside_down)

    if frame_number:
        text = f"Frame: {frame_number} | Centroid: ({cx}, {cy}),{w}x{h} pixels"
    else:
        text = f"Centroid: ({cx}, {cy}),{w}x{h} pixels"

    annotate_image(base_image,text,text_size=1.5)
    annotate_image(rectified_frame, text)

    if debug:
        show_bgr(base_image, title=f"Detected Object, Frame {frame_number}",
                 w=debug_image_width)
        show_bgr(rectified_frame, title=f"Rectified Frame, Frame {frame_number}",
                 w=debug_image_width)
                 

    return rectified_frame,base_image, cx,cy,w,h, contour_mask

