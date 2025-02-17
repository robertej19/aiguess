import cv2, os, math, glob
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.common_tools import annotate_image, show_bgr
from utils.detection_tools import  extract_contour_region, create_lab_range_mask, expand_mask

def detect_basic(frame_to_process,frame_number=None,debug=False,
                   ip_params = None,save_figs=False,
                    debug_image_width = 14,
                    b_range = None,
                    o_range = None):
    
    raw_frame = frame_to_process.copy()
    frame_to_return = frame_to_process.copy()

    if b_range is not None:
        if o_range is not None:
            # enable color processing
            min_vals_extended = np.array([b_range[0][0]-100, b_range[0][1]-100, b_range[0][2]-30])
            max_vals_extended = np.array([b_range[1][0]+100, b_range[1][1]+100, b_range[1][2]+30])
            print("new min max",min_vals_extended,max_vals_extended)
            m_b = create_lab_range_mask(raw_frame, min_vals_extended, max_vals_extended)

            big_b = expand_mask(m_b, kernel_size=20)


            min_vals_extended = o_range[0]-50
            max_vals_extended = o_range[1]+50
            print("min max",min_vals_extended,max_vals_extended)
            m_o= create_lab_range_mask(raw_frame, min_vals_extended, max_vals_extended)

            combined_mask = cv2.bitwise_and(big_b, m_o)
            #apply combined_mask to raw_frame
            raw_frame = cv2.bitwise_and(raw_frame, raw_frame, mask=big_b)

            
    
    # if pass a frame with no context, assume there is 1 easy to detect object in the frame, and write to yaml
    if ip_params is None:
        print("Warning: No ImageProcessingParams object was passed. Using default values.")
        adaptive_threshold_max_value = 255
        adaptive_threshold_blockSize  = 53
        adaptive_threshold_constant   = 51.5579384567383
        sobel_pre_gaussian_kernel = [3,3]
        sobel_pre_gaussian_sigma  = 0.5
        sobel_x_kernel = 3
        sobel_y_kernel = 3
        sobel_threshold = 50
        lab_offset = 10
        object_w_max_threshold = 18
        object_h_max_threshold = 4.729192443183128
    else:
        adaptive_threshold_max_value = ip_params.adaptive_threshold_max_value
        adaptive_threshold_blockSize = ip_params.adaptive_threshold_blockSize
        adaptive_threshold_constant = ip_params.adaptive_threshold_constant
        sobel_pre_gaussian_kernel = ip_params.sobel_pre_gaussian_kernel
        sobel_pre_gaussian_sigma  = ip_params.sobel_pre_gaussian_sigma
        sobel_x_kernel = ip_params.sobel_x_kernel
        sobel_y_kernel = ip_params.sobel_y_kernel
        sobel_threshold = ip_params.sobel_threshold
        object_w_max_threshold = ip_params.object_w_max_threshold
        object_h_max_threshold = ip_params.object_h_max_threshold
        
    if debug:
        show_bgr(raw_frame, title=f"Raw Frame {frame_number}",
                    w=debug_image_width,
                    save_fig=save_figs,save_fig_name=f"frame_{frame_number}/0_raw_frame.png")
        
 
    gray_raw_frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)

    blurred_gray_frame = cv2.GaussianBlur(gray_raw_frame, 
                                        sobel_pre_gaussian_kernel, 
                                        sobel_pre_gaussian_sigma)
    if debug:
        show_bgr(blurred_gray_frame, title=f"Pre-Adaptive Threshold, Frame {frame_number}",
                    w=debug_image_width,
                    save_fig=save_figs,save_fig_name=f"frame_{frame_number}/1_blurred_gray_frame.png")

        
    adaptive_thresh = cv2.adaptiveThreshold(
        blurred_gray_frame,
        maxValue=adaptive_threshold_max_value,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=adaptive_threshold_blockSize,
        C=adaptive_threshold_constant
    )


    if debug:
        show_bgr(adaptive_thresh, title=f"Adaptive Threshold, Frame {frame_number}",
                    w=debug_image_width,
                    save_fig=save_figs,save_fig_name=f"frame_{frame_number}/2_adaptive_thresh.png")


    # Sobel
    sobel_x = cv2.Sobel(adaptive_thresh, cv2.CV_64F, 1, 0, ksize=sobel_x_kernel)
    sobel_y = cv2.Sobel(adaptive_thresh, cv2.CV_64F, 0, 1, ksize=sobel_y_kernel)
    sobel_mag = cv2.magnitude(sobel_x, sobel_y)
    sobel_abs = cv2.convertScaleAbs(sobel_mag)

    if debug:
        show_bgr(sobel_abs, title=f"Sobel Abs, Frame {frame_number}",
                 w=debug_image_width)

    # Threshold
    _, sobel_edges = cv2.threshold(sobel_abs, sobel_threshold, 
                                  255, cv2.THRESH_BINARY)


    if debug:
        show_bgr(sobel_edges, title=f"Edges in Frame, Frame {frame_number}",
                 w=debug_image_width,
                    save_fig=save_figs,save_fig_name=f"frame_{frame_number}/3_sobel_edges.png")


    filtered_contours = None
    identified_object = None

    contours, _ = cv2.findContours(sobel_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if debug:
        # Draw all contours
        frame_with_contours = raw_frame.copy()
        cv2.drawContours(frame_with_contours, contours, -1, (0, 255, 0), 1)
        show_bgr(frame_with_contours, title=f"All Contours, Frame {frame_number}",
                 w=debug_image_width,
                    save_fig=save_figs,save_fig_name=f"frame_{frame_number}/4_all_contours.png")

    if contours:
        # Filter contours to exclude those with an area larger than 400 pixels
        # filtered_contours = [contour for contour in contours if cv2.contourArea(contour) <= object_area_threshold]
        # instead of filtering on area, filter on min and max w and h
        filtered_contours = [contour for contour in contours if cv2.boundingRect(contour)[2] <= object_w_max_threshold and cv2.boundingRect(contour)[3] <= object_h_max_threshold]

    if debug:
        # Draw filtered contours
        frame_with_filtered_contours = raw_frame.copy()
        cv2.drawContours(frame_with_filtered_contours, filtered_contours, -1, (0, 255, 0), 1)
        show_bgr(frame_with_filtered_contours, title=f"Filtered Contours, Frame {frame_number}",
                 w=debug_image_width,
                    save_fig=save_figs,save_fig_name=f"frame_{frame_number}/5_filtered_contours.png")

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
        

        extracted, contour_mask = extract_contour_region(raw_frame, identified_object)

        #Zoom in on the object
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        # Apply the closing operation.
        closed = cv2.morphologyEx(sobel_edges, cv2.MORPH_CLOSE, kernel)

        # Optionally, find contours on the closed image:
        contours, hierarchy = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw the closed contour on a copy of the original image for visualization
        output = cv2.cvtColor(sobel_edges, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(output, contours, -1, (0, 255, 0), 2)

        if debug:
            show_bgr(output, title=f"Closed Contours, Frame {frame_number}",
                     w=debug_image_width,
                    save_fig=save_figs,save_fig_name=f"frame_{frame_number}/6_closed_contour_object.png")
        zoomed_in = raw_frame[y-2:y+h+2, x-2:x+w+2]

        if debug:
            show_bgr(zoomed_in, title=f"Zoomed In, Frame {frame_number}",
                     w=debug_image_width,
                    save_fig=save_figs,save_fig_name=f"frame_{frame_number}/7_zoomed_in.png")
            
            show_bgr(extracted, title=f"Extracted Region, Frame {frame_number}",
                     w=debug_image_width,
                    save_fig=save_figs,save_fig_name=f"frame_{frame_number}/8_extracted.png")
            
        cv2.rectangle(frame_to_return, (x-2*w, y-2*h), (x+4*w, y+4*h), (0, 0, 255), 4)
        cv2.drawContours(frame_to_return, [identified_object], -1, (0, 255, 0), 1)

    else:
        contour_mask = None
        identified_object = None
        cx,cy,w,h = 0,0,0,0
    
    if frame_number:
        text = f"Frame: {frame_number} | Centroid: ({cx}, {cy}),{w}x{h} pixels"
    else:
        text = f"Centroid: ({cx}, {cy}),{w}x{h} pixels"

    annotate_image(frame_to_return,text,text_size=1.5)

    if debug:
        show_bgr(frame_to_return, title=f"Detected Object, Frame {frame_number}",
                 w=debug_image_width,
                    save_fig=save_figs,save_fig_name=f"frame_{frame_number}/9_detected_object.png")
                 

    return frame_to_return, cx,cy,w,h, contour_mask, identified_object

