import cv2, os, math, glob
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from skimage import feature
from sklearn.linear_model import LinearRegression
from skimage.transform import rotate,warp

from utils.detection_tools import detect_sky, estimate_horizon_line_by_edges
from utils.detection_tools import rectify_horizon, downsampler, rotate_and_center_horizon
from utils.common_tools import annotate_image, show_bgr

from utils.common_tools import find_nonzero_bounding_box, trim_video, draw_parallel_lines


def find_birds(raw_frame,frame_number=None):
    
    base_image = raw_frame.copy()

    ## Detection is added by having an initial Gaussian blur
    ## Check that this is actually used
    base_image = cv2.GaussianBlur(base_image,(3,3),1)

    very_blurred_base_image = cv2.GaussianBlur(base_image, (125, 125), 125)
    sky_mask = detect_sky(very_blurred_base_image)



    scale_factor = 1
    
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
    cv2.line(base_image, (x_start, y_start), (x_end, y_end), (0, 0, 255), 1)
    
    base_image_x = base_image.copy()
    cv2.line(base_image, (x_start*scale_factor, y_start*scale_factor), (x_end*scale_factor, y_end*scale_factor), (0, 0, 255), 5)
    

    double_line, region_mask, upside_down =  draw_parallel_lines(base_image_x,sky_mask, slope, intercept* scale_factor, distance=horizon_search_offset)

    #show_bgr(region_mask, title="Region Mask {frame_number}")
    #show_bgr(double_line)

    #show_bgr(masked_frame_sky_roi)
    gray_frame = cv2.cvtColor(base_image, cv2.COLOR_BGR2GRAY)


    # Read an image (can be color or already grayscale)
    input_image = base_image.copy()
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

    hsv_frame = raw_frame.copy()

    hsv_frame_hsv =cv2.cvtColor(hsv_frame, cv2.COLOR_BGR2HSV)
    hb = 110
    sb = 128
    vb = 255
    ht = 130
    st = 250
    vt = 255
    b = [hb,sb,vb]
    t = [ht,st,vt]
    lb = np.array(b, np.uint8)
    ub = np.array(t, np.uint8)
    hsv_mask = cv2.inRange(hsv_frame_hsv, lb, ub)


    
    
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
    edges_in_sky_roi_with_hsv_mask = cv2.bitwise_and(edges_in_sky_roi,hsv_mask)


    #show_bgr(edges_in_sky_roi, title=f"Region Mask {frame_number}")

    #show_bgr(edges_in_sky_roi)
    filtered_contours = None
    contours, _ = cv2.findContours(edges_in_sky_roi_with_hsv_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Filter contours to exclude those with an area larger than 400 pixels
        filtered_contours = [contour for contour in contours if cv2.contourArea(contour) <= 400]

    if filtered_contours:
        # Find the largest remaining contour
        largest_contour = max(filtered_contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        print(x,y,w,h)
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
        cv2.rectangle(base_image, (x-2*w, y-2*h), (x+4*w, y+4*h), (0, 0, 255), 4)
        #show_bgr(base_image, title=f"Contour Mask {frame_number}")

        to_rect = base_image.copy()
        rectified_frame = rotate_and_center_horizon(to_rect,slope,intercept*scale_factor,upside_down=upside_down)


        if frame_number:
            text = f"Frame: {frame_number} | Centroid: ({cx}, {cy}),{w}x{h} pixels"
        else:
            text = f"Centroid: ({cx}, {cy}),{w}x{h} pixels"
        annotate_image(base_image, text,text_size=1.5)
        annotate_image(rectified_frame, text)


        return rectified_frame,base_image, cx,cy,w,h
    else:
        if frame_number:
            text = f"Frame: {frame_number} | Centroid: ({0}, {0}), Size {0}x{0} pixels"
        else:
            text = f"Centroid: ({0}, {0}), Size: {0}x{0} pixels"

        annotate_image(base_image,text,text_size=1.5)
        rectified_frame = rotate_and_center_horizon(base_image,slope,intercept*scale_factor)
        annotate_image(rectified_frame, text)

        
        return rectified_frame, base_image, 0,0,0,0

