import cv2
import numpy as np

def upscale_frame(frame: np.ndarray, target_width: int = 1920, target_height: int = 1080) -> np.ndarray:
    """
    Upscales the given frame to the specified target size.

    Parameters:
        frame (np.ndarray): The input image/frame.
        target_width (int): The desired output width (default is 1920).
        target_height (int): The desired output height (default is 1080).

    Returns:
        np.ndarray: The upscaled frame.
    """
    # Use cv2.resize to upscale the frame
    upscaled_frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_CUBIC)
    return upscaled_frame

# Example usage:
if __name__ == "__main__":
    # Load an image from disk (replace 'input.jpg' with your image file)
    frame = cv2.imread('input.jpg')
    
    if frame is not None:
        upscaled = upscale_frame(frame)
        # Save the upscaled image
        cv2.imwrite('upscaled_1920x1080.jpg', upscaled)
        # Optionally, display the image in a window
        cv2.imshow('Upscaled Frame', upscaled)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Error: Could not load the image.")
