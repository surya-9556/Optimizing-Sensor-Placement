import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import shutil
import cv2
import math
import numpy as np

from scipy.ndimage import gaussian_filter
from sklearn.preprocessing import StandardScaler

class VideoGeneration:
    
    def frames_generation(self, sample_dataset, output_dir, column):
        sample_dataset[column] = sample_dataset[column].astype(float)

        num_sensors = len(sample_dataset['Sensor ID'].unique())
        range_values = math.ceil(len(sample_dataset) / num_sensors)

        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)

        os.makedirs(output_dir)

        frames = []
        for i in range(1, range_values + 1):
            temp_dataset = sample_dataset.groupby('Sensor ID').nth(i - 1).reset_index()
            temp_dataset['Sensor ID'] = temp_dataset['Sensor ID'].astype('category').cat.codes

            sensor_id_matrix = temp_dataset[column].values.reshape(8, num_sensors // 8)

            plt.figure(figsize=(10, 6))
            sns.heatmap(sensor_id_matrix, annot=False, cmap="YlGnBu", fmt=".2f", cbar=False, xticklabels=False, yticklabels=False)
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

            output_file = os.path.join(output_dir, f"heatmap_{i}.png")
            plt.savefig(output_file)
            plt.close()

            frames.append(output_file)

        return frames
    
    def video_generation(self, frames, video_output):
        frame = cv2.imread(frames[0])
        height, width, _ = frame.shape

        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        video = cv2.VideoWriter(video_output, fourcc, 24.0, (width, height))

        prev_frame = cv2.imread(frames[0])

        for frame_path in frames[1:]:
            next_frame = cv2.imread(frame_path)
            
            for i in range(1, 10):
                alpha = i / 10.0
                interpolated_frame = cv2.addWeighted(prev_frame, 1 - alpha, next_frame, alpha, 0)
                video.write(interpolated_frame)

            prev_frame = next_frame

        video.release()
        cv2.destroyAllWindows()

        return f"Video saved as: {video_output}"
    
    def reconstructed_humidity_frames_generation(self, output_dir, reduced_df_sorted, data, column):
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)

        os.makedirs(output_dir)

        frames = []
        sensor_id_to_filtered_value = []  # List to store (sensor_id, filtered_value) tuples

        for i in range(1, 101):
            temp_dataset = reduced_df_sorted.groupby('Sensor ID').nth(i - 1).reset_index()

            # Prepare the matrix
            rows, cols = 8, 7
            sensor_matrix = np.zeros((rows, cols), dtype=float)

            sensor_positions = {}  # To map sensor_id to their matrix position
            for sensor_id, (row, col) in data.items():
                sensor_positions[sensor_id] = (rows - row, col - 1)
                if sensor_id in temp_dataset['Sensor ID'].astype(str).values:
                    temp_value = temp_dataset[temp_dataset['Sensor ID'].astype(str) == sensor_id][column].values
                    if len(temp_value) > 0:
                        sensor_matrix[rows - row, col - 1] = temp_value[0]

            # Apply Gaussian filter
            sigma = min(max(np.std(sensor_matrix), 1), 5)
            smoothed_data = gaussian_filter(sensor_matrix, sigma=sigma, mode='nearest')

            # Replace zeros with smoothed values
            filled_data = np.where(sensor_matrix == 0, smoothed_data, sensor_matrix)

            # Map filtered values back to sensor_ids
            for sensor_id, (row, col) in sensor_positions.items():
                filtered_value = filled_data[row, col]
                sensor_id_to_filtered_value.append({'Sensor ID': sensor_id, 'Filtered Value': filtered_value})

            # Plot the heatmap
            plt.figure(figsize=(10, 6))
            sns.heatmap(filled_data, annot=False, cmap="YlGnBu", fmt=".2f", cbar=False, xticklabels=False, yticklabels=False)
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

            output_file = os.path.join(output_dir, f"heatmap_{i}.png")
            plt.savefig(output_file)
            plt.close()

            frames.append(output_file)

        # Convert the list of sensor_id to filtered_value mappings to DataFrame
        sensor_id_filtered_df = pd.DataFrame(sensor_id_to_filtered_value)

        return frames, sensor_id_filtered_df

    def reconstrucing_frames(self, model, not_reduced_df_sorted, reduced_df_sorted, output_dir, data,column):

        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)

        os.makedirs(output_dir)

        frames = []
        all_data = []

        for i in range(1, 101):
            predicting_data = not_reduced_df_sorted.groupby('Sensor ID',observed=False).nth(i - 1)[['Sensor ID',column]].reset_index()
            predicted_data_array = predicting_data[column].values.reshape(1, -1)

            scaler1 = StandardScaler()
            predicted_scaled = scaler1.fit_transform(predicted_data_array)

            predictions = model.predict(predicted_scaled)

            predicted_data = pd.DataFrame({"Sensor ID": predicting_data['Sensor ID'].unique(),
                                    column:predictions[0]})
            
            sampled_Actual_data = reduced_df_sorted.groupby('Sensor ID').nth(i - 1)[['Sensor ID',column]]
            merged_df = pd.merge(predicted_data,sampled_Actual_data,on=['Sensor ID',column], how='outer')

            merged_df['Sensor ID'] = pd.Categorical(merged_df['Sensor ID'], categories=list(data.keys()), ordered=True)
            merged_df_sorted = merged_df.sort_values(by='Sensor ID')
            merged_df_sorted = merged_df_sorted[~pd.isna(merged_df_sorted['Sensor ID'])]
            merged_df_sorted['Sensor ID'] = merged_df_sorted['Sensor ID'].astype(str)
            merged_df_sorted.reset_index(drop=True, inplace=True)

            # Reshape the DataFrame to an 8x7 matrix
            matrix = merged_df_sorted[column].values.reshape(8, 7)

            # Plot the heatmap
            plt.figure(figsize=(10, 6))
            sns.heatmap(matrix, annot=False, fmt=".2f", cmap='YlGnBu', cbar=False, xticklabels=False, yticklabels=False)
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
            # plt.show()
            output_file = os.path.join(output_dir, f"heatmap_{i}.png")
            plt.savefig(output_file)
            plt.close()

            frames.append(output_file)

            all_data.append(merged_df_sorted)
        
        all_data = pd.concat(all_data,ignore_index=True)

        return frames, all_data