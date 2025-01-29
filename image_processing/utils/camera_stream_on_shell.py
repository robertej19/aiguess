import cv2, os, time, shutil, sys
import numpy as np

# Default ASCII chars for black-and-white
BW_ASCII_CHARS = ["@", "#", "S", "%", "?", "*", "+", ";", ":", ",", "."]
# Simplified color ASCII chars (all '#')
COLOR_ASCII_CHARS = ["#"] * 11

############################
# Helper: Convert a pixel to ASCII (black-and-white)
############################
def pixel_to_ascii_bw(r, g, b, ascii_chars):
    # Convert to brightness
    # Numbers chosen just because work well
    brightness = 0.2126 * r + 0.7152 * g + 0.0722 * b 
    index = int((brightness / 255) * (len(ascii_chars) - 1))
    return ascii_chars[index]

############################
# Helper: Convert a pixel to ASCII with color (ANSI)
############################
def pixel_to_ascii_color(r, g, b):
    # For color mode, we ignore brightness-based variation.
    # We'll always use '#', tinted by the pixel's color.
    # ASCII_CHARS is just a repeated '#' for bigger blocks.
    ansi_char = '#'
    # 24-bit color code: \033[38;2;R;G;Bm
    return f"\033[38;2;{r};{g};{b}m{ansi_char}\033[0m"

############################
# Convert a frame to ASCII text lines
############################
def frame_to_ascii(frame, new_width=80, color=False):
    # Convert from BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    h, w, _ = rgb_frame.shape
    aspect_ratio = h / w

    # Approx correction factor for text aspect ratio
    # This helps squares not look squashed vertically
    new_height = int(aspect_ratio * new_width * 0.55)

    # Resize the frame
    resized = cv2.resize(rgb_frame, (new_width, new_height))

    lines = []

    if color:
        # Use the repeated '#' array for blocky color.
        for row in resized:
            line_chars = []
            for (r, g, b) in row:
                line_chars.append(pixel_to_ascii_color(r, g, b))
            lines.append("".join(line_chars))
    else:
        # Black-and-white ASCII
        for row in resized:
            line_chars = []
            for (r, g, b) in row:
                line_chars.append(pixel_to_ascii_bw(r, g, b, BW_ASCII_CHARS))
            lines.append("".join(line_chars))

    # Join lines with newlines
    ascii_frame = "\n".join(lines)
    return ascii_frame


############################
# Main function
############################
def display_ascii_video(
    source=0,
    use_color=False,
    width=None,
    fps=10
):
    # Attempt to pick default width from terminal size if not specified.
    if width is None:
        try:
            term_size = shutil.get_terminal_size()
            width = term_size.columns
        except:
            width = 80

    # Prepare video capture
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Cannot open video source {source}")
        return

    # figure out real fps if available, fallback to parameter if needed
    real_fps = cap.get(cv2.CAP_PROP_FPS)
    if real_fps <= 0 or real_fps > 60_000:
        real_fps = fps

    # Slow down video on purpose
    # real_fps = 10

    # Prepare video writers
    # We'll record the real video frames and the ASCII frames.
    timestamp = int(time.time() * 1000)
    real_video_name = f"cam_video_{timestamp}.mp4"

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    real_writer = cv2.VideoWriter(real_video_name, fourcc, real_fps, (frame_width, frame_height))

    # create a dummy black frame for measuring
    ret, test_frame = cap.read()
    if not ret:
        print("Error: Could not read a single frame from the source.")
        cap.release()
        return
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # reset to start

    # Cross-platform clear command
    clear_cmd = "cls" if os.name == "nt" else "clear"

    print("Recording real video to:", real_video_name)
    print("Press Ctrl+C to exit...")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Write real video frame
            real_writer.write(frame)

            # Generate ASCII text
            ascii_text = frame_to_ascii(frame, new_width=width, color=use_color)

            # Display ASCII in terminal
            os.system(clear_cmd)
            print(ascii_text)

            #its already slow enough, can probably remove this:
            time.sleep(1 / fps)

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        # Release everything
        cap.release()
        real_writer.release()

        print("Video writers closed. Exiting...")

if __name__ == "__main__":
    # Example usage:
    #   python script.py 0 True 80 10
    # where arguments are: <source> <use_color> <width> <fps>

    print("Usage: python askme.py <source> <use_color> <width> <fps>")
    print("Example: python askme.py 0 True 80 10")
    print("  source: Video source (default: 0 for webcam)")
    print("  use_color: Use color ASCII (default: True)")
    print("  width: ASCII width in chars (default: terminal width). Make this small to go faster")
    print("  fps: Frames per second (default: 30)")



    # Basic parse from sys.argv
    source = 0
    color_mode = True
    width = None
    fps = 30


    if len(sys.argv) > 1:
        source = sys.argv[1]
        # If it's a digit, cast to int
        if source.isdigit():
            source = int(source)
    if len(sys.argv) > 2:
        color_mode = (sys.argv[2].lower() == 'true')
    if len(sys.argv) > 3:
        width = int(sys.argv[3])
    if len(sys.argv) > 4:
        fps = float(sys.argv[4])

    display_ascii_video(
        source=source,
        use_color=color_mode,
        width=width,
        fps=fps
    )
