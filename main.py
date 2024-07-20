import json
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
import zipfile
from Functions import video_generation, extracting_files, neural_networks

video_gen = video_generation.VideoGeneration()
extract_files = extracting_files.ExtractingFiles()
neural_network = neural_networks.NeuralNetwork()

placement_json = os.getenv('placement_json')
column_name = os.getenv('column_name')

with open(placement_json, 'r') as file:
    data = json.load(file)

keys_list = list(data.keys())

class OptimizingPlacement:
    @staticmethod
    def extracting_files(file_name, sensor_data):

        if file_name.endswith('.zip'):
            extract_files.extract_zip(file_name, sensor_data)

            initial_data = extract_files.read_all_excel_files(sensor_data)
        else:
            
            initial_data = extract_files.read_all_excel_files(file_name)

        df_sorted = extract_files.sorting_data(initial_data, keys_list)

        combined_data = df_sorted.drop_duplicates(subset=['Time', column_name])

        return combined_data, keys_list
    
    @staticmethod
    def heatmap_video_gen(combined_data, output_dir, video_output):
        sample_dataset = combined_data.groupby(column_name).apply(lambda x: x.tail(100)).reset_index(drop=True)
        sample_dataset = extract_files.sorting_data(sample_dataset, keys_list)

        frames = video_gen.frames_generation(sample_dataset, output_dir)

        video_gen.video_generation(frames, video_output)

        return "Successfully generated the heatmap video."
    
    @staticmethod
    def replacing_missing_values(combined_data):
        if combined_data['Temperature'].isnull().sum() > 0 or combined_data['Humidity'].isnull().sum() > 0:
            if combined_data['Temperature'].isnull().sum() > 0:
                combined_data = neural_network.train_and_fill(combined_data, 'Temperature', 100)
            if combined_data['Humidity'].isnull().sum() > 0:
                combined_data = neural_network.train_and_fill(combined_data, 'Humidity', 100)
        else:
            print('No null values detected!!!!!!')

        return combined_data
