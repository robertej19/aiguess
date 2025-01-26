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

def find_birds(raw_frame):
    
    base_image = raw_frame
    image = raw_frame#[400:1800, 1100:1500]
    scale_factor = 4
    sky_mask = detect_sky(raw_frame)
    downsampled_sky_mask = downsampler(sky_mask,scale_factor=scale_factor)
    
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
    
    
    cv2.line(raw_frame, (x_start*scale_factor, y_start*scale_factor), (x_end*scale_factor, y_end*scale_factor), (0, 0, 255), 5)
    base_image_rgb = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
    
    
    #rectified_frame = rectify_horizon(raw_frame, slope, intercept*scale_factor)

    
    height, width = raw_frame.shape[:2]
    
    top_left = (0,0)
    top_right = (width,0)
    bottom_left = (x_start*scale_factor, max(0,.85*y_start*scale_factor-30))
    bottom_right =  (x_end*scale_factor, max(0,.85*y_end*scale_factor-30))
    
    roi_corners = np.array([
        [top_left[0],      top_left[1]],
        [top_right[0],     top_right[1]],
        [bottom_right[0],  bottom_right[1]],
        [bottom_left[0],   bottom_left[1]]
    ], dtype=np.int32)
    
    polygon_roi_mask = np.zeros((height, width), dtype=np.uint8)
    
    cv2.fillPoly(polygon_roi_mask, [roi_corners], 255)
    polygon_roi_mask = create_line_roi_mask_vectorized(width, height, slope, .8*scale_factor*intercept, above=True)
    masked_frame_sky_roi = cv2.bitwise_and(raw_frame, raw_frame, mask=polygon_roi_mask)
    
    #show_bgr(masked_frame_sky_roi)
    gray_roi = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
    kernel_size=(3, 3)
    sigma=0.5
    gaussian_blur = cv2.GaussianBlur(gray_roi, kernel_size, sigma)


    sobel_threshold=50
    #canny_edges = cv2.Canny(gray_roi, threshold1=50, threshold2=150)
    sobelx = cv2.Sobel(gray_roi, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray_roi, cv2.CV_64F, 0, 1, ksize=3)
    sobel = cv2.magnitude(sobelx, sobely)
    sobel = cv2.convertScaleAbs(sobel)
    
    
    _, sobel_mask = cv2.threshold(sobel, sobel_threshold, 255, cv2.THRESH_BINARY)
    
    edges_in_roi = cv2.bitwise_and(sobel_mask, polygon_roi_mask)
    # -- Optional: Morphological operations (e.g., close small gaps) --
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # -- 3. Find contours in the binary mask --
    contours, _ = cv2.findContours(edges_in_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
        cv2.rectangle(base_image, (x, y), (x+w, y+h), (0, 255, 0), 4)

        rectified_frame = rotate_and_center_horizon(base_image,slope,intercept*scale_factor)


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

