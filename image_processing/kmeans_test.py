import cv2
import numpy as np

def kmeans_segment_and_describe(frame, K=2):
    """
    Performs K-means clustering on the input 'frame' (BGR) with K=2,
    identifies which cluster is background vs. object using a size heuristic,
    then returns:
      - A visualization image with the object bounding box drawn.
      - A dictionary of properties for each cluster.
    
    Args:
        frame (np.ndarray): Input image in BGR format.
        K (int): Number of clusters (default=2).

    Returns:
        vis_bgr (np.ndarray): The original frame with a bounding box around the smaller cluster.
        properties (dict): Dictionary containing the following keys:
            "background": {
                "cluster_id": int,
                "pixel_count": int,
                "average_bgr": (float, float, float)
            },
            "object": {
                "cluster_id": int,
                "pixel_count": int,
                "average_bgr": (float, float, float),
                "bounding_box": (x_min, y_min, x_max, y_max),
                "centroid": (cx, cy)
            }
    """
    # 1) Convert the frame from BGR to Lab
    lab_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    
    # 2) Reshape the Lab image to a 2D array of [L, A, B] pixels
    H, W, _ = frame.shape
    pixel_vals = lab_frame.reshape((-1, 3))
    
    # 3) Convert to float32 for kmeans
    pixel_vals = np.float32(pixel_vals)

    # 4) K-means parameters
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    
    # 5) Run K-means
    _, labels, centers = cv2.kmeans(pixel_vals, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # labels is of shape (H*W, 1); reshape to the 2D image
    labels_2d = labels.reshape((H, W))
    
    # 6) Identify the two clusters (0 and 1)
    # Count how many pixels belong to each cluster
    mask_cluster0 = (labels_2d == 0).astype(np.uint8)
    mask_cluster1 = (labels_2d == 1).astype(np.uint8)
    
    count_cluster0 = np.sum(mask_cluster0)
    count_cluster1 = np.sum(mask_cluster1)
    
    # 7) Decide which cluster is "background" vs "object"
    # Here we assume the larger cluster = background, smaller = object
    if count_cluster0 > count_cluster1:
        background_id, object_id = 0, 1
        background_mask, object_mask = mask_cluster0, mask_cluster1
        background_count, object_count = count_cluster0, count_cluster1
    else:
        background_id, object_id = 1, 0
        background_mask, object_mask = mask_cluster1, mask_cluster0
        background_count, object_count = count_cluster1, count_cluster0
    
    # 8) Calculate average BGR for each cluster
    # Use cv2.mean(...) with a mask to get the mean color in BGR space
    # First, we need to do it on the original frame
    mean_bgr_bg = cv2.mean(frame, mask=background_mask)[:3]  # first 3 elements => (B, G, R)
    mean_bgr_obj = cv2.mean(frame, mask=object_mask)[:3]
    
    # 9) Compute bounding box and centroid for the object cluster
    #    We'll locate all (y, x) coords where object_mask==1
    y_indices, x_indices = np.where(object_mask == 1)
    
    if len(x_indices) > 0 and len(y_indices) > 0:
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        
        # Centroid
        cx = np.mean(x_indices)
        cy = np.mean(y_indices)
        
        bounding_box = (int(x_min), int(y_min), int(x_max), int(y_max))
        centroid = (cx, cy)
    else:
        # No object found (or no pixels in that cluster)
        bounding_box = (0, 0, 0, 0)
        centroid = (0, 0)
    
    # 10) Prepare the dictionary of properties
    props = {
        "background": {
            "cluster_id": background_id,
            "pixel_count": int(background_count),
            "average_bgr": tuple(mean_bgr_bg)
        },
        "object": {
            "cluster_id": object_id,
            "pixel_count": int(object_count),
            "average_bgr": tuple(mean_bgr_obj),
            "bounding_box": bounding_box,
            "centroid": centroid
        }
    }
    
    # 11) Create a visualization
    # Copy original image to draw bounding box
    vis_bgr = frame.copy()
    
    # Draw bounding box around the smaller cluster (object)
    (x_min, y_min, x_max, y_max) = bounding_box
    if x_max > x_min and y_max > y_min:
        cv2.rectangle(vis_bgr, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    
    # Optionally, draw centroid
    cv2.circle(vis_bgr, (int(centroid[0]), int(centroid[1])), 5, (0, 0, 255), -1)
    
    return vis_bgr, props

# --- Example usage ---
if __name__ == "__main__":
    # Load a sample image (in BGR)
    img = cv2.imread("frame_of_interest.jpg")
    
    # Segment with K=2
    result_img, props = kmeans_segment_and_describe(img, K=2)
    
    # Print the properties
    print("=== Cluster Properties ===")
    print("Background:")
    print("  cluster_id:", props["background"]["cluster_id"])
    print("  pixel_count:", props["background"]["pixel_count"])
    print("  average_bgr:", props["background"]["average_bgr"])
    
    print("\nObject:")
    print("  cluster_id:", props["object"]["cluster_id"])
    print("  pixel_count:", props["object"]["pixel_count"])
    print("  average_bgr:", props["object"]["average_bgr"])
    print("  bounding_box:", props["object"]["bounding_box"])
    print("  centroid:", props["object"]["centroid"])
    
    # Show the result with big size
    cv2.imshow("Original", img)
    cv2.imshow("K=2 Segmentation + Object Bounding Box", result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
