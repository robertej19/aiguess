import cv2

import os
import time
import shutil
import sys
import numpy as np
import matplotlib.pyplot as plt

from utils.detection_tools import extract_object_and_background_masks
from utils.detection_tools import get_min_max_hsv
from utils.common_tools import numbered_framing_from_ascii



def main():
    """
    Usage:
      python zoom_ascii.py my_video.mp4
    Interaction:
      - For each frame, we display ASCII
      - The user can:
         (s) Zoom in again on the same frame by specifying row1 col1 row2 col2 in ASCII coords
         (n) Move to the next frame
         (p) Play the rest of the video without interruption
         (q) Quit
    """
    if len(sys.argv) < 2:
        print("Usage: python zoom_ascii.py <video_file>")
        sys.exit(1)

    video_path = sys.argv[1]
    #cap = cv2.VideoCapture(video_path)
    cap = cv2.VideoCapture(0) #use usb cam

    if not cap.isOpened():
        print(f"Error: Cannot open video file '{video_path}'")
        sys.exit(1)

    frame_index = 5  # start at frame 5 for debugging, else 0
    play_continuously = False
    playcounter = 30
    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if not ret:
            print("End of video or cannot read frame.")
            break

        frame_index += 1
        print(f"--- Frame {frame_index} ---")

        current_frame = frame.copy()
        global_roi = (0, 0, frame.shape[0], frame.shape[1])

        while True:
            if not play_continuously:
                os.system("cls" if os.name == "nt" else "clear")
                edges = cv2.Canny(current_frame, 10, 20)
                ascii_text, (ascii_h, ascii_w) = numbered_framing_from_ascii(
                    current_frame,
                    new_width=256,
                    color=True,
                    enumerate_grid=True
                )
                print(ascii_text)
                print("\nOptions: [s] zoom on the same frame, [n] next frame, [p] play continuously, [d] to detect object, [q] quit")
                choice = input("> ").lower().strip()

                if choice == 'q':
                    cap.release()
                    print("Quitting...")
                    return None

                elif choice == 'd':
                    cap.release()
                    print("Quitting...")
                    return current_frame

                elif choice == 'n':
                    break

                elif choice == 'p':
                    play_continuously = True
                    break

                elif choice == 's':
                    print("Enter ROI in ASCII coords: row1 col1 row2 col2 (or press Enter to cancel)")
                    line = input("Coordinates: ").strip()
                    if not line:
                        continue

                    parts = line.split()
                    if len(parts) != 4:
                        print("Invalid input. Must enter 4 numbers, e.g. '10 20 30 40'")
                        continue

                    try:
                        r1_ascii, c1_ascii, r2_ascii, c2_ascii = map(int, parts)
                    except ValueError:
                        print("Invalid input. Must be integers.")
                        continue

                    current_h, current_w = current_frame.shape[:2]
                    scale_y = current_h / ascii_h
                    scale_x = current_w / ascii_w

                    r1_crop = int(r1_ascii * scale_y)
                    c1_crop = int(c1_ascii * scale_x)
                    r2_crop = int(r2_ascii * scale_y)
                    c2_crop = int(c2_ascii * scale_x)

                    if r2_crop < r1_crop:
                        r1_crop, r2_crop = r2_crop, r1_crop
                    if c2_crop < c1_crop:
                        c1_crop, c2_crop = c2_crop, c1_crop

                    r1_crop = max(0, min(r1_crop, current_h - 1))
                    r2_crop = max(0, min(r2_crop, current_h))
                    c1_crop = max(0, min(c1_crop, current_w - 1))
                    c2_crop = max(0, min(c2_crop, current_w))

                    if (r2_crop - r1_crop) < 1 or (c2_crop - c1_crop) < 1:
                        print("ROI is too small or invalid.")
                        continue

                    current_frame = current_frame[r1_crop:r2_crop, c1_crop:c2_crop]
                    print(f"Cropped to: row=({r1_crop},{r2_crop}), col=({c1_crop},{c2_crop})")
                    continue

                else:
                    print("Invalid choice. Please type 's', 'n', 'p', or 'q'.")
            else:
                os.system("cls" if os.name == "nt" else "clear")

                if playcounter > 0:
                    playcounter -= 1
                    ascii_text, (ascii_h, ascii_w) = numbered_framing_from_ascii(
                        current_frame,
                        new_width=256,
                        color=True,
                        enumerate_grid=False
                    )
                    print(ascii_text)
                else:

                    hsv_lower_bound = [50,100 , 135]    # (hb, sb, vb)
                    hsv_upper_bound = [80, 225, 255]  # (ht, st, vt)

                    hsv_frame_hsv =cv2.cvtColor(current_frame, cv2.COLOR_BGR2HSV)

                    lb = np.array(hsv_lower_bound, np.uint8)
                    ub = np.array(hsv_upper_bound, np.uint8)
                    hsv_mask = cv2.inRange(hsv_frame_hsv, lb, ub)

                    contours, _ = cv2.findContours(hsv_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    frame = np.zeros_like(current_frame)
                    
                    if contours:
                        # Find the largest remaining contour
                        identified_object = max(contours, key=cv2.contourArea)
                        x, y, w, h = cv2.boundingRect(identified_object)
                        bounding_box = (x, y, w, h)
        
                        #create an empty frame
                        #draw the identified object
                        cv2.drawContours(frame, [identified_object], -1, (0, 255, 0), 2)
                        #put a red dot on the center
                        cv2.circle(frame, (int(x + w/2), int(y + h/2)), 3, (0, 0, 255), -1)

                    ascii_text, (ascii_h, ascii_w) = numbered_framing_from_ascii(
                        frame,
                        new_width=256,
                        color=True,
                        enumerate_grid=False
                    )
                    print(ascii_text)
                #pause
                time.sleep(0.05)
                break

    cap.release()
    print("Finished all frames.")



if __name__ == "__main__":

    frame_of_interest = None
    #frame_of_interest = main()
    
    if frame_of_interest is not None:
        # save frame of interest
        cv2.imwrite("frame_of_interest.jpg", frame_of_interest)

        
        # do something with the frame
        print("Frame of interest shape:", frame_of_interest.shape)
        cv2.imshow("Frame", frame_of_interest)
        key = cv2.waitKey(0)
        cv2.destroyAllWindows()

        # apply gaussian blur
        blurred = cv2.GaussianBlur(frame_of_interest, (3, 3), 0)
        cv2.imshow("Blurred", blurred)
        key = cv2.waitKey(0)
        cv2.destroyAllWindows()

        # apply edge detection
        edges = cv2.Canny(blurred, 100, 200)
        cv2.imshow("Edges", edges)
        key = cv2.waitKey(0)
        cv2.destroyAllWindows()



        os.system("cls" if os.name == "nt" else "clear")
        ascii_text, (ascii_h, ascii_w) = numbered_framing_from_ascii(
            edges,
            new_width=128,
            color=True,
            enumerate_grid=True
        )
        print(ascii_text)
    if frame_of_interest is None:
        # load frame of interest
        frame_of_interest = cv2.imread("frame_of_interest.jpg")

    object_mask, background_mask = extract_object_and_background_masks(frame_of_interest)
    cv2.imshow("Object Mask", object_mask)
    cv2.imshow("Background Mask", background_mask)
    key = cv2.waitKey(0)
    cv2.destroyAllWindows()    
    output = get_min_max_hsv(frame_of_interest, object_mask)
    print(output)
    # Example usage:
    # frame = cv2.imread("example.jpg")  # Load an image
    # obj_mask, bg_mask = extract_object_and_background_masks(frame)
    # cv2.imshow("Object Mask", obj_mask)
    # cv2.imshow("Background Mask", bg_mask)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
