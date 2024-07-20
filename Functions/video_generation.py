import os
import matplotlib.pyplot as plt
import seaborn as sns
import shutil
import cv2
import math

class VideoGeneration:
    
    def frames_generation(self, sample_dataset, output_dir):
        sample_dataset['Temperature'] = sample_dataset['Temperature'].astype(float)

        num_sensors = len(sample_dataset['Sensor ID'].unique())
        range_values = math.ceil(len(sample_dataset) / num_sensors)

        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)

        os.makedirs(output_dir)

        frames = []
        for i in range(1, range_values + 1):
            temp_dataset = sample_dataset.groupby('Sensor ID').nth(i - 1).reset_index()
            temp_dataset['Sensor ID'] = temp_dataset['Sensor ID'].astype('category').cat.codes

            sensor_id_matrix = temp_dataset['Temperature'].values.reshape(8, num_sensors // 8)

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
