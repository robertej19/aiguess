import cv2

import os
import time
import shutil
import sys
import numpy as np
import matplotlib.pyplot as plt

#import select
#import termios
#import tty

# Default ASCII chars for black-and-white
BW_ASCII_CHARS = ["@", "#", "S", "%", "?", "*", "+", ";", ":", ",", "."]
# Simplified color ASCII chars (all '#')
COLOR_ASCII_CHARS = ["#"] * 11

def pixel_to_ascii_bw(r, g, b, ascii_chars):
    brightness = 0.2126 * r + 0.7152 * g + 0.0722 * b 
    index = int((brightness / 255) * (len(ascii_chars) - 1))
    return ascii_chars[index]

def pixel_to_ascii_color(r, g, b):
    ansi_char = '#'
    return f"\033[38;2;{r};{g};{b}m{ansi_char}\033[0m"

def frame_to_ascii(frame, new_width=80, color=False, enumerate_grid=False):
    """
    Convert a BGR frame to ASCII text lines. 
    If enumerate_grid=True, we print row/col labels in the top-left corner:
      - Top 3 rows show column digits (hundreds, tens, ones).
      - Left 2 columns show row digits (tens, ones).
    Returns:
      ascii_text (str): The ASCII-art representation
      (ascii_height, ascii_width): the shape of the ASCII image
    """
    # Convert BGR -> RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, _ = rgb_frame.shape

    # Compute new height based on aspect ratio
    aspect_ratio = h / w
    new_height = int(aspect_ratio * new_width * 0.5625)
    if new_height < 1:
        new_height = 1

    # Resize
    resized = cv2.resize(rgb_frame, (new_width, new_height))

    ascii_lines = []
    for i, row in enumerate(resized):
        line_chars = []
        for j, (r, g, b) in enumerate(row):
            if enumerate_grid:
                # Show row/col indices in top-left
                if i < 3:
                    # top 3 rows used to show column digits
                    if i == 0:
                        line_chars.append(str(j // 100))
                    elif i == 1:
                        line_chars.append(str((j % 100) // 10))
                    else:
                        line_chars.append(str(j % 10))
                elif j < 2:
                    # left 2 columns used to show row digits
                    if j == 0:
                        line_chars.append(str(i // 10))
                    else:
                        line_chars.append(str(i % 10))
                else:
                    # normal pixel to ascii
                    if color:
                        line_chars.append(pixel_to_ascii_color(r, g, b))
                    else:
                        line_chars.append(pixel_to_ascii_bw(r, g, b, BW_ASCII_CHARS))
            else:
                # normal ASCII
                if color:
                    line_chars.append(pixel_to_ascii_color(r, g, b))
                else:
                    line_chars.append(pixel_to_ascii_bw(r, g, b, BW_ASCII_CHARS))
        ascii_lines.append("".join(line_chars))

    ascii_text = "\n".join(ascii_lines)
    return ascii_text, resized.shape[:2]  # (height, width)


def main():
    """
    Usage:
      python zoom_ascii.py my_video.mp4
    Interaction:
      - For each frame, we display ASCII
      - The user can:
         (s) Zoom in again on the same frame by specifying row1 col1 row2 col2 in ASCII coords
         (n) Move to the next frame
         (q) Quit
    """
    if len(sys.argv) < 2:
        print("Usage: python zoom_ascii.py <video_file>")
        sys.exit(1)

    video_path = sys.argv[1]
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file '{video_path}'")
        sys.exit(1)

    # We'll display each frame, let the user zoom, then move on.
    frame_index = 5 # start at frame 45 for debugging, else #0
    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if not ret:
            print("End of video or cannot read frame.")
            break

        frame_index += 1
        print(f"--- Frame {frame_index} ---")

        # We'll keep the user's current crop in `roi` (r1,c1,r2,c2) in the *original* frame coords
        # We'll do a repeated zoom on the same frame until user chooses next frame or quit.
        current_frame = frame.copy()
        # show frame

        # Store the offset from the original top-left if we keep zooming in multiple times
        # So that final ROI is always in the full-frame coordinate system
        global_roi = (0, 0, frame.shape[0], frame.shape[1])  # the entire frame

        while True:
            # 1) Convert the current (cropped) frame to ASCII
            #    with row/col enumeration so user can pick bounding box.
            os.system("cls" if os.name == "nt" else "clear")
            edges = cv2.Canny(current_frame, 10, 20)

            ascii_text, (ascii_h, ascii_w) = frame_to_ascii(
                current_frame,
                new_width=256,
                color=True,
                enumerate_grid=True
            )
            print(ascii_text)

            # 2) Ask user what to do next
            print("\nOptions: [s] zoom on the same frame, [n] next frame, [q] quit")
            choice = input("> ").lower().strip()

            if choice == 'q':
                # quit everything
                cap.release()
                print("Quitting...")
                return None

            elif choice == 'd':
                # proceed to next frame
                cap.release()
                print("Quitting...")
                return current_frame

            elif choice == 'n':
                # proceed to next frame
                break  # exit this "zoom loop" and read next frame from cap


            elif choice == 's':
                # let user pick ROI in ASCII coords
                print("Enter ROI in ASCII coords: row1 col1 row2 col2 (or press Enter to cancel)")
                line = input("Coordinates: ").strip()
                if not line:
                    continue  # user didn't type anything, just ignore

                parts = line.split()
                if len(parts) != 4:
                    print("Invalid input. Must enter 4 numbers, e.g. '10 20 30 40'")
                    continue

                try:
                    r1_ascii, c1_ascii, r2_ascii, c2_ascii = map(int, parts)
                except ValueError:
                    print("Invalid input. Must be integers.")
                    continue

                # Convert ASCII coords to the coordinate system of current_frame
                current_h, current_w = current_frame.shape[:2]
                scale_y = current_h / ascii_h
                scale_x = current_w / ascii_w

                # Convert
                r1_crop = int(r1_ascii * scale_y)
                c1_crop = int(c1_ascii * scale_x)
                r2_crop = int(r2_ascii * scale_y)
                c2_crop = int(c2_ascii * scale_x)

                # Sort them in case user typed reversed corners
                if r2_crop < r1_crop:
                    r1_crop, r2_crop = r2_crop, r1_crop
                if c2_crop < c1_crop:
                    c1_crop, c2_crop = c2_crop, c1_crop

                # Make sure they're within bounds
                r1_crop = max(0, min(r1_crop, current_h - 1))
                r2_crop = max(0, min(r2_crop, current_h))
                c1_crop = max(0, min(c1_crop, current_w - 1))
                c2_crop = max(0, min(c2_crop, current_w))

                if (r2_crop - r1_crop) < 1 or (c2_crop - c1_crop) < 1:
                    print("ROI is too small or invalid.")
                    continue

                # Now crop the 'current_frame' to that region
                current_frame = current_frame[r1_crop:r2_crop, c1_crop:c2_crop]

                #cv2.imshow("Frame", current_frame)
                #key = cv2.waitKey(0)
                #cv2.destroyAllWindows()

                # show the current real frame
                print(f"Cropped to: row=({r1_crop},{r2_crop}), col=({c1_crop},{c2_crop})")

                # We remain on this frame, so user can choose to zoom again, or go next
                continue

            else:
                print("Invalid choice. Please type 's', 'n', or 'q'.")

    # If we exit the while True loop, we are done reading frames
    cap.release()
    print("Finished all frames.")


if __name__ == "__main__":
    frame_of_interest = main()
    if frame_of_interest is not None:
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
        ascii_text, (ascii_h, ascii_w) = frame_to_ascii(
            edges,
            new_width=256,
            color=True,
            enumerate_grid=True
        )
        print(ascii_text)


