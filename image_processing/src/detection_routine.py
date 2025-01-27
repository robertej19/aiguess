import cv2, os, math, glob
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from skimage import feature
from sklearn.linear_model import LinearRegression
from skimage.transform import rotate,warp

from utils.detection_tools import detect_sky, estimate_horizon_line_by_edges, create_line_roi_mask_vectorized
from utils.detection_tools import rectify_horizon, downsampler, translate_vertical, rotate_and_center_horizon
from utils.common_tools import annotate_image, show_bgr

from utils.common_tools import find_nonzero_bounding_box, trim_video, draw_parallel_lines


def find_birds(raw_frame):
    base_image = raw_frame.copy()
    sky_mask = detect_sky(raw_frame)


    scale_factor = 4
    downsampled_sky_mask = downsampler(sky_mask,scale_factor=scale_factor)
    
    half = downsampled_sky_mask.shape[0] // 2
    sum_top = np.sum(downsampled_sky_mask[:half, :])     # top half
    sum_bottom = np.sum(downsampled_sky_mask[half:, :])  # bottom half

    # If more sky is in the bottom, flip vertically
    upside_down = False
    if sum_bottom > sum_top:
        upside_down = True
        #output_image = cv2.flip(output_image, 0)  # 0 => flip vertically


    result = estimate_horizon_line_by_edges(downsampled_sky_mask)

    if result is None:
        #raise ValueError("estimate_horizon_line_by_edges() returned None. Check the input or the function logic.")
        slope,intercept = 0, downsampled_sky_mask.shape[0]/2
    else:
        slope, intercept = result

    h,w = downsampled_sky_mask.shape
    
    x_start, x_end = 0, w - 1
    y_start = int(slope * x_start + intercept)
    y_end   = int(slope * x_end   + intercept)
    #############################################
    ### SKY ROI CREATED
    ###########################################
    cv2.line(downsampled_sky_mask, (x_start, y_start), (x_end, y_end), (0, 0, 255), 1)
    
    raw_frame_x = raw_frame.copy()
    cv2.line(raw_frame, (x_start*scale_factor, y_start*scale_factor), (x_end*scale_factor, y_end*scale_factor), (0, 0, 255), 5)
    base_image_rgb = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
    

    double_line, region_mask =  draw_parallel_lines(raw_frame_x,sky_mask, slope, intercept* scale_factor, distance=50)

    #show_bgr(double_line)

    #show_bgr(masked_frame_sky_roi)
    gray_frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)


    # Read an image (can be color or already grayscale)
    input_image = raw_frame.copy()
    # Convert to grayscale if needed
    gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

    # Adaptive Threshold
    # - maxValue = 255
    # - adaptiveMethod = cv2.ADAPTIVE_THRESH_GAUSSIAN_C (or cv2.ADAPTIVE_THRESH_MEAN_C)
    # - thresholdType = cv2.THRESH_BINARY
    # - blockSize = 11 (size of neighborhood, must be odd)
    # - C = 2 (constant subtracted from the mean or weighted mean)
    adaptive_thresh = cv2.adaptiveThreshold(
        gray,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=5,
        C=4
    )

    #show_bgr(adaptive_thresh)
    # Apply some Gaussian Blur
    gaussian_blur_ksize = (3, 3)
    gaussian_sigma = 0.5
    blurred_gray = cv2.GaussianBlur(adaptive_thresh, gaussian_blur_ksize, gaussian_sigma)

    # Sobel
    sobel_x = cv2.Sobel(blurred_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blurred_gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = cv2.magnitude(sobel_x, sobel_y)
    sobel_abs = cv2.convertScaleAbs(sobel_mag)

    # Threshold
    sobel_threshold = 50
    _, sobel_mask = cv2.threshold(sobel_abs, sobel_threshold, 255, cv2.THRESH_BINARY)

    # Restrict edges to sky ROI
    edges_in_sky_roi = cv2.bitwise_and(sobel_mask, region_mask)

    #show_bgr(edges_in_sky_roi)

    contours, _ = cv2.findContours(edges_in_sky_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        bounding_box = (x, y, w, h)
        
        # Centroid using image moments
        M = cv2.moments(largest_contour)
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
        cv2.rectangle(base_image, (x, y), (x+w, y+h), (0, 0, 255), 8)

        to_rect = base_image.copy()
        rectified_frame = rotate_and_center_horizon(to_rect,slope,intercept*scale_factor,upside_down=upside_down)


        #cv2.circle(roi, (cx, cy), 4, (0, 0, 255), -1)
        text = f"Detected Centroid: ({cx}, {cy}), Size: {w}x{h} pixels"
        annotate_image(base_image, text)
        annotate_image(rectified_frame, text)


        return rectified_frame,base_image, cx,cy,w,h
    else:
        text = f"Detected Centroid: ({0}, {0}), Size: {0}x{0} pixels"

        annotate_image(base_image,text)
        rectified_frame = rotate_and_center_horizon(base_image,slope,intercept*scale_factor)
        annotate_image(rectified_frame, text)

        
        return rectified_frame, base_image, 0,0,0,0

