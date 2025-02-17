import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

import cv2,sys,os
import matplotlib.pyplot as plt
import cv2, os, math, glob
import matplotlib.pyplot as plt
import numpy as np

os.sys.path.append(os.getcwd())


from utils.common_tools import annotate_image, show_bgr
from utils.common_tools import stack_videos, assemble_frames_to_video
from utils.common_tools import find_nonzero_bounding_box, trim_video, draw_parallel_lines

from utils.detection_tools import detect_sky, estimate_horizon_line_by_edges
from utils.detection_tools import  downsampler, rotate_and_center_horizon
from utils.common_tools import annotate_image, show_bgr, draw_parallel_lines
from src.detect_basic import detect_basic
from src.analysis_routine import calculate_errors



import numpy as np
import random
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque

# --- Assume the external detection function is available ---
# For example:
# from my_detection_module import detect_basic
# detect_basic should have the following signature:
#   _, detected_x, detected_y, _, _, _, _ = detect_basic(frame, 100, debug=False, ip_params=params, save_figs=False,
#                                                         debug_image_width=14, b_range=None, o_range=None)

import numpy as np
import random
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque

import numpy as np
import cv2

# --- Assume that the external detection function is available ---
# For example:
# from my_detection_module import detect_basic
# The signature should be similar to:
#   _, detected_x, detected_y, _, _, _, _ = detect_basic(frame, 100, debug=False, ip_params=params, save_figs=False,
#                                                         debug_image_width=14, b_range=None, o_range=None)

# --- Define the ImageProcessingParams class ---
class ImageProcessingParams:
    def __init__(self, 
                 adaptive_threshold_max_value=255, 
                 adaptive_threshold_blockSize=5, 
                 adaptive_threshold_constant=4,
                 sobel_pre_gaussian_kernel=(3,3),
                 sobel_pre_gaussian_sigma=0.5,
                 sobel_x_kernel=3,
                 sobel_y_kernel=3,
                 sobel_threshold=50,
                 lab_offset=10,
                 object_w_max_threshold=20,
                 object_h_max_threshold=20):
        self.adaptive_threshold_max_value = adaptive_threshold_max_value
        self.adaptive_threshold_blockSize = adaptive_threshold_blockSize
        self.adaptive_threshold_constant = adaptive_threshold_constant
        self.sobel_pre_gaussian_kernel = sobel_pre_gaussian_kernel
        self.sobel_pre_gaussian_sigma = sobel_pre_gaussian_sigma
        self.sobel_x_kernel = sobel_x_kernel
        self.sobel_y_kernel = sobel_y_kernel
        self.sobel_threshold = sobel_threshold
        self.lab_offset = lab_offset
        self.object_w_max_threshold = object_w_max_threshold
        self.object_h_max_threshold = object_h_max_threshold

    def __str__(self):
        return (f"adaptive_threshold_max_value: {self.adaptive_threshold_max_value}, "
                f"adaptive_threshold_blockSize: {self.adaptive_threshold_blockSize}, "
                f"adaptive_threshold_constant: {self.adaptive_threshold_constant}, "
                f"sobel_pre_gaussian_kernel: {self.sobel_pre_gaussian_kernel}, "
                f"sobel_pre_gaussian_sigma: {self.sobel_pre_gaussian_sigma}, "
                f"sobel_x_kernel: {self.sobel_x_kernel}, "
                f"sobel_y_kernel: {self.sobel_y_kernel}, "
                f"sobel_threshold: {self.sobel_threshold}, "
                f"lab_offset: {self.lab_offset}, "
                f"object_w_max_threshold: {self.object_w_max_threshold}, "
                f"object_h_max_threshold: {self.object_h_max_threshold}")

# --- Define an environment class that evaluates a candidate ---
class ImageProcessingTuningEnvBinary:
    def __init__(self, image_path='resources/foi_2b.png', epsilon=5):
        self.image_path = image_path
        self.true_x = 556# 33 4 3497
        self.true_y = 33#457
        self.epsilon = epsilon  # success radius in pixels
        self.frame = cv2.imread(self.image_path)
        if self.frame is None:
            raise ValueError(f"Image not found at {self.image_path}")
        # Fixed parameters (for those not tuned)
        self.fixed_params = {
            'adaptive_threshold_max_value': 255,
            'sobel_pre_gaussian_kernel': (3,3),
            'sobel_pre_gaussian_sigma': 0.5,
            'sobel_x_kernel': 3,
            'sobel_y_kernel': 3,
            'sobel_threshold': 50,
            'lab_offset': 10
        }
    
    def evaluate(self, action):
        """
        Given a normalized 4-dimensional action, scale it to real parameter values,
        run detect_basic, and return a binary reward.
        """
        # Scale parameters:
        object_w_max_threshold = action[0] * 50          # [0,50]
        object_h_max_threshold = action[1] * 50          # [0,50]
        # Map [0,1] to odd integer between 3 and 15:
        maxblocksize = 155
        blockSize = int(2 * round(action[2] * maxblocksize/2) + 3)
        constant = action[3] * maxblocksize                       # [0,10]
        #make sure constant is less than blocksize
        constant = min(constant, blockSize-1)

        params = ImageProcessingParams(
            adaptive_threshold_max_value = self.fixed_params['adaptive_threshold_max_value'],
            adaptive_threshold_blockSize = blockSize,
            adaptive_threshold_constant = constant,
            sobel_pre_gaussian_kernel = self.fixed_params['sobel_pre_gaussian_kernel'],
            sobel_pre_gaussian_sigma = self.fixed_params['sobel_pre_gaussian_sigma'],
            sobel_x_kernel = self.fixed_params['sobel_x_kernel'],
            sobel_y_kernel = self.fixed_params['sobel_y_kernel'],
            sobel_threshold = self.fixed_params['sobel_threshold'],
            lab_offset = self.fixed_params['lab_offset'],
            object_w_max_threshold = object_w_max_threshold,
            object_h_max_threshold = object_h_max_threshold
        )
        
        # Run the external detection routine:
        result = detect_basic(self.frame, 100, debug=False, ip_params=params, save_figs=False,
                              debug_image_width=14, b_range=None, o_range=None)
        detected_x = result[1]
        detected_y = result[2]
        # Compute Euclidean distance:
        dist = np.sqrt((detected_x - self.true_x)**2 + (detected_y - self.true_y)**2)
        # Binary reward: 1 if within epsilon, else 0.
        reward = 1 if dist <= self.epsilon else 0
        return reward
    
def evolution_strategy(env, num_iterations=100, pop_size=50, sigma=0.1, learning_rate=0.02):
    """
    Simple Evolution Strategy (ES) optimizer.
    
    If any candidate receives a reward of 1, it immediately returns that candidate.
    
    Args:
      env: Environment with an evaluate(action) method that returns a binary reward.
      num_iterations: Number of generations.
      pop_size: Number of candidates per generation.
      sigma: Standard deviation for the noise.
      learning_rate: Step size for the update.
      
    Returns:
      A normalized 4-dimensional parameter vector (values in [0,1]) that produced a reward of 1,
      or the best found candidate after all iterations.
    """
    # Initialize candidate solution (theta) as a 4-d vector in [0,1]. Start at 0.5 for each.
    theta = np.full(4, 0.5)
    best_reward = 0
    best_theta = theta.copy()
    
    for iteration in range(num_iterations):
        noise = np.random.randn(pop_size, 4)
        rewards = np.zeros(pop_size)
        
        # Evaluate each candidate.
        for i in range(pop_size):
            candidate = theta + sigma * noise[i]
            candidate = np.clip(candidate, 0, 1)
            reward = env.evaluate(candidate)
            rewards[i] = reward
            
            # Immediately return if any candidate achieves the binary reward (i.e. reward == 1)
            if reward == 1:
                print(f"Solution found at iteration {iteration}, candidate index {i} with reward {reward}.")
                return candidate
        
        baseline = np.mean(rewards)
        # Compute gradient estimate (centered rewards)
        grad = np.dot(noise.T, rewards - baseline) / pop_size
        
        theta = theta + learning_rate * grad
        theta = np.clip(theta, 0, 1)
        
        avg_reward = np.mean(rewards)
        if avg_reward > best_reward:
            best_reward = avg_reward
            best_theta = theta.copy()
        
        print(f"Iteration {iteration:03d}: avg reward = {avg_reward:.3f}, best so far = {best_reward:.3f}")
    
    return best_theta

# Example usage (assuming your environment class is defined as before):
if __name__ == "__main__":
    env = ImageProcessingTuningEnvBinary(image_path='resources/foi_2b.png', epsilon=5)
    best_theta = evolution_strategy(env, num_iterations=200, pop_size=50, sigma=0.25, learning_rate=0.02)
    print("Best normalized parameters:", best_theta)
    
    # Map the normalized parameters to real parameter values.
    best_object_w = best_theta[0] * 50
    best_object_h = best_theta[1] * 50
    maxblocksize = 155
    best_blockSize = int(2 * round(best_theta[2] * maxblocksize/2) + 3)
    best_constant = best_theta[3] * maxblocksize                       # [0,10]
    #make sure constant is less than blocksize
    constant = min(best_constant, best_blockSize-1)

    best_params = ImageProcessingParams(
        adaptive_threshold_max_value = 255,
        adaptive_threshold_blockSize = best_blockSize,
        adaptive_threshold_constant = best_constant,
        sobel_pre_gaussian_kernel = (3,3),
        sobel_pre_gaussian_sigma = 0.5,
        sobel_x_kernel = 3,
        sobel_y_kernel = 3,
        sobel_threshold = 50,
        lab_offset = 10,
        object_w_max_threshold = best_object_w,
        object_h_max_threshold = best_object_h
    )
    print("Best learned ImageProcessingParams:")
    print(best_params)

    #run image detection with best params
    result = detect_basic(env.frame, 100, debug=True, ip_params=best_params, save_figs=False,
                          debug_image_width=14, b_range=None, o_range=None)
    detected_x = result[1]
    detected_y = result[2]
    print(f"Detected: ({detected_x}, {detected_y})")