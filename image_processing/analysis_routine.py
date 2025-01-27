
import pandas as pd
import matplotlib.pyplot as plt

# 1. Read in the data
# Replace 'frame_data.txt' and 'processed_data.txt' with your actual file paths if needed
truth_df = pd.read_csv('synth_videos/horizon_2/frame_data.txt')        # Columns: frame_number, local_dot_x_truth, local_dot_y_truth
estimate_df = pd.read_csv('synth_videos/horizon_2/processed_data.txt') # Columns: frame_index, cx, cy

# 2. Merge data on the matching frame_number/frame_index
merged_df = pd.merge(
    truth_df,
    estimate_df,
    left_on='frame_number',
    right_on='frame_index',
    how='inner'
)

# 3. Rename columns for clarity
merged_df.rename(
    columns={
        'local_dot_x_truth': 'truth_cx',
        'local_dot_y_truth': 'truth_cy',
        'cx': 'estimated_cx',
        'cy': 'estimated_cy'
    },
    inplace=True
)

# If you want to keep only the required columns:
merged_df = merged_df[['frame_index', 'truth_cx', 'truth_cy', 'estimated_cx', 'estimated_cy']]

# 4. Calculate Euclidean distance
merged_df['distance'] = np.sqrt(
    (merged_df['truth_cx'] - merged_df['estimated_cx'])**2 +
    (merged_df['truth_cy'] - merged_df['estimated_cy'])**2
)

# 5. Plot frame_index on x-axis vs. distance on y-axis
plt.figure(figsize=(10, 6))
plt.plot(merged_df['frame_index'], merged_df['distance'], marker='o')
plt.title('Distance between Truth and Estimated Points')
plt.xlabel('Frame Index')
plt.ylabel('Euclidean Distance')
plt.grid(True)
plt.show()

# 6. (Optional) Inspect the merged data
print(merged_df.head())
