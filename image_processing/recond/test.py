import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# -----------------------------------------------------------
# Part 1: Simulate the ball's 3D trajectory (as before)
# -----------------------------------------------------------
dt = 0.01          # time step (seconds)
g = 9.81           # gravitational acceleration (m/s^2)

# Initial conditions: starting at the origin with an initial velocity
initial_position = np.array([0.0, 0.0, 0.0])
initial_velocity = np.array([10.0, 0, 20.0])  # m/s in (x, y, z)

positions = []     # will hold the ball's positions over time
velocity = initial_velocity.copy()
position = initial_position.copy()

# Simulate until the ball hits the ground (z <= 0)
while position[2] >= 0:
    positions.append(position.copy())
    position += velocity * dt
    velocity[2] -= g * dt  # only the z-component is affected by gravity

positions = np.array(positions)

# -----------------------------------------------------------
# Part 2: Set up the camera in the world and project points
# -----------------------------------------------------------
# Define the camera's extrinsic properties:
#   - Camera position in world coordinates
camera_position = np.array([25, -50.0, 20.0])  # placed at (x=50, y=50, z=20)
#   - Define a "look-at" target point (choose a point in the scene)
target = np.array([0.0, 0.0, 10.0])
#   - Define the "up" vector in the world (here, the z-axis is up)
up = np.array([0.0, 0.0, 1.0])

def look_at(eye, target, up):
    """
    Create a rotation matrix from world to camera coordinates.
    This function assumes a camera coordinate system where:
      - The z-axis points from the camera toward the target.
      - The x-axis points to the right.
      - The y-axis points down (when combined with the image coordinate convention later).
    """
    # Compute the forward (z) axis
    z_axis = target - eye
    z_axis = z_axis / np.linalg.norm(z_axis)
    # Compute the right (x) axis
    x_axis = np.cross(z_axis, up)
    x_axis = x_axis / np.linalg.norm(x_axis)
    # Compute the true up (y) axis
    y_axis = np.cross(x_axis, z_axis)
    # Stack as rows: this rotation matrix transforms (world_point - eye) into camera coordinates.
    R = np.vstack([x_axis, y_axis, z_axis])
    return R

# Compute the rotation matrix for the camera
R = look_at(camera_position, target, up)

# Define the camera's intrinsic parameters:
f = 400        # focal length in pixels
cx = 320       # principal point x-coordinate (image center)
cy = 240       # principal point y-coordinate (image center)
K = np.array([[f, 0, cx],
              [0, f, cy],
              [0, 0,  1]])

# Define image dimensions (in pixels)
img_width, img_height = 640, 480

def project_point(world_point):
    """
    Project a 3D world point into the camera's image plane.
    Returns:
      - A tuple (proj, depth) where:
          proj: a 2D point [x, y] in pixel coordinates if the point is in front of the camera (None otherwise)
          depth: the z-coordinate in the camera coordinate system (None if behind the camera)
    """
    # Transform the world point into the camera coordinate system.
    # X_cam = R * (X_world - camera_position)
    cam_point = R.dot(world_point - camera_position)
    
    # Check if the point is in front of the camera (positive depth)
    if cam_point[2] <= 0:
        return None, None
    
    # Perspective projection: divide by depth then apply the intrinsic matrix
    proj_homog = K.dot(cam_point / cam_point[2])
    x_img, y_img = proj_homog[0], proj_homog[1]
    return np.array([x_img, y_img]), cam_point[2]

# Pre-compute the projection for each ball position along with its depth.
proj_data = []  # Each element is a tuple: (projected_point, depth)
for pos in positions:
    proj, depth = project_point(pos)
    proj_data.append((proj, depth))

# -----------------------------------------------------------
# Part 3: Animate the camera's view with variable ball size
# -----------------------------------------------------------
fig, ax = plt.subplots()
ax.set_xlim(0, img_width)
ax.set_ylim(0, img_height)
# If desired, uncomment the next line to follow typical image coordinates where y=0 is at the top.
# ax.invert_yaxis()  
ax.set_title("Camera View of the Ball")
ax.set_xlabel("Pixel X")
ax.set_ylabel("Pixel Y")

# Represent the ball as a red dot.
ball_dot, = ax.plot([], [], 'ro', markersize=8)

# Define the physical radius of the ball in meters.
ball_radius = 0.5

def init():
    ball_dot.set_data([], [])
    return ball_dot,

def update(frame):
    proj, depth = proj_data[frame]
    # If the ball is in front of the camera:
    if proj is not None:
        ball_dot.set_data(proj[0], proj[1])
        # Compute the projected marker size.
        # The projected radius in pixels is given by: (f * ball_radius) / depth.
        # Here we use that value as the marker size (in points).
        marker_size = (f * ball_radius) / depth
        ball_dot.set_markersize(marker_size)
    else:
        ball_dot.set_data([], [])
    return ball_dot,

ani = FuncAnimation(fig, update, frames=len(proj_data),
                    init_func=init, blit=True, interval=10)

plt.show()
