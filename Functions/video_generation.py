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
        for i in range(1, 101):
            temp_dataset = reduced_df_sorted.groupby('Sensor ID').nth(i - 1).reset_index()

            # Prepare the matrix
            rows, cols = 8, 7
            sensor_matrix = np.zeros((rows, cols), dtype=float)  # Assuming temperature is a float

            # Populate the matrix
            for sensor_id, (row, col) in data.items():
                if sensor_id in temp_dataset['Sensor ID'].astype(str).values:
                    temp_value = temp_dataset[temp_dataset['Sensor ID'].astype(str) == sensor_id][column].values
                    if len(temp_value) > 0:
                        sensor_matrix[rows - row, col - 1] = temp_value[0]

            # Apply Gaussian filter
            # sigma = 2  # Standard deviation for Gaussian kernel
            sigma = min(max(np.std(sensor_matrix), 1), 5)
            smoothed_data = gaussian_filter(sensor_matrix, sigma=sigma, mode='nearest')

            # Replace zeros with smoothed values
            filled_data = np.where(sensor_matrix == 0, smoothed_data, sensor_matrix)

            plt.figure(figsize=(10, 6))
            sns.heatmap(filled_data, annot=False, cmap="YlGnBu", fmt=".2f", cbar=False, xticklabels=False, yticklabels=False)
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

            output_file = os.path.join(output_dir, f"heatmap_{i}.png")
            plt.savefig(output_file)
            plt.close()

            frames.append(output_file)

        return frames
    
    def reconstructed_frames_generation(self, non_reduced_sensors, X_scaled, model, Optional_data, data, output_dir,column):

        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)

        os.makedirs(output_dir)

        frames = []

        for i in range(1,101):
            # 3. Prediction
            predicted_data = non_reduced_sensors.groupby('Sensor ID',observed=False).nth(i - 1)[['Sensor ID',column]].reset_index()

            # Ensure that the 'Temperature' column is an array of shape (25,)
            predicted_data_array = predicted_data[column].values.reshape(1, -1)

            # Fit the scaler using the predicted_data_array to maintain consistency
            scaler1 = StandardScaler()
            predicted_scaled = scaler1.fit_transform(predicted_data_array)

            # Pad the input to match the model's input shape
            num_sensors = X_scaled.shape[1]
            padding_length = num_sensors - predicted_scaled.shape[1]
            padded_input = np.pad(predicted_scaled, ((0, 0), (0, padding_length)), mode='constant')

            # Make predictions
            predictions = model.predict(padded_input)

            # Remove the padded part to keep the original length
            predictions_trimmed = predictions[:, :predicted_data_array.shape[1]]

            predicted_data = pd.DataFrame({"Sensor ID": predicted_data['Sensor ID'].unique(),
                        column:predictions_trimmed[0]})
            # print(predicted_data)
            sample_data = Optional_data.groupby('Sensor ID',observed=False).nth(i - 1)[['Sensor ID',column]].reset_index()
            sample_data = sample_data.drop('index',axis=1)

            # Merge the DataFrames on 'Sensor ID'
            merged_df = pd.merge(predicted_data, sample_data, on=['Sensor ID',column], how='outer')
            merged_df['Sensor ID'] = pd.Categorical(merged_df['Sensor ID'], categories=list(data.keys()), ordered=True)
            reduced_df_sorted = merged_df.sort_values(by='Sensor ID')
            reduced_df_sorted = reduced_df_sorted[~pd.isna(reduced_df_sorted['Sensor ID'])]
            reduced_df_sorted['Sensor ID'] = reduced_df_sorted['Sensor ID'].astype(str)
            reduced_df_sorted.reset_index(drop=True, inplace=True)

            # Reshape the DataFrame to an 8x7 matrix
            matrix = reduced_df_sorted[column].values.reshape(8, 7)

            # Plot the heatmap
            plt.figure(figsize=(10, 6))
            sns.heatmap(matrix, annot=False, fmt=".2f", cmap='YlGnBu', cbar=False, xticklabels=False, yticklabels=False)
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

            output_file = os.path.join(output_dir, f"heatmap_{i}.png")
            plt.savefig(output_file)
            plt.close()

            frames.append(output_file)

        return frames
