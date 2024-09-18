import json
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
import zipfile
from Functions import video_generation, extracting_files, neural_networks

# Initialize instances of custom classes from the imported modules
video_gen = video_generation.VideoGeneration()
extract_files = extracting_files.ExtractingFiles()
neural_network = neural_networks.NeuralNetwork()

# Retrieve environment variables for placement JSON file and column name
placement_json = os.getenv('placement_json')
column_name = os.getenv('column_name')

# Load the JSON file containing keys
with open(placement_json, 'r') as file:
    data = json.load(file)

# Extract the list of keys from the JSON data
keys_list = list(data.keys())

class OptimizingPlacement:
    @staticmethod
    def extracting_files(file_name = None, sensor_data = None):
        """
        Extracts files from a ZIP archive or directly reads Excel files.
        The function sorts and combines data, removing duplicates based on 'Time' and a specified column.

        Parameters:
        - file_name: Name of the file (ZIP or Excel).
        - sensor_data: Directory or path to extract files to.

        Returns:
        - combined_data: DataFrame with sorted and deduplicated data.
        - keys_list: List of keys extracted from the JSON data.
        """

        if file_name == None:
            # If it's not a ZIP, read the Excel file directly
            initial_data = extract_files.read_all_excel_files(sensor_data)

        elif file_name.endswith('.zip'):
            # If the file is a ZIP archive, extract it
            extract_files.extract_zip(file_name, sensor_data)
            # Read all Excel files from the extracted contents
            initial_data = extract_files.read_all_excel_files(sensor_data)
            
        elif file_name.endswith('.xlsx'):
            initial_data = extract_files.read_all_excel_files(file_name)

        # Sort the data based on the keys list
        df_sorted = extract_files.sorting_data(initial_data, keys_list)

        # Remove duplicates based on 'Time' and the specified column
        combined_data = df_sorted.drop_duplicates(subset=['Time', column_name])

        return combined_data, keys_list
    
    @staticmethod
    def heatmap_video_gen(combined_data, output_dir, video_output, column):
        """
        Generates a heatmap video from the combined data.

        Parameters:
        - combined_data: DataFrame with combined and processed data.
        - output_dir: Directory to save the generated frames.
        - video_output: Path to save the generated video.
        - column: The column to group data by.

        Returns:
        - A success message upon generating the video.
        """

        # Sample the dataset by taking the last 100 rows of each group defined by column_name
        sample_dataset = combined_data.groupby(column_name).apply(lambda x: x.tail(100)).reset_index(drop=True)
        # Sort the sampled data again based on the keys list
        sample_dataset = extract_files.sorting_data(sample_dataset, keys_list)

        # Generate frames for the video from the sampled dataset
        frames = video_gen.frames_generation(sample_dataset, output_dir, column)

        # Create the video using the generated frames
        video_gen.video_generation(frames, video_output)

        return "Successfully generated the heatmap video."
    
    @staticmethod
    def replacing_missing_values(combined_data):
        """
        Replaces missing values in the 'Temperature' and 'Humidity' columns using a neural network model.

        Parameters:
        - combined_data: DataFrame with combined and processed data.

        Returns:
        - combined_data: DataFrame with missing values replaced.
        """

        # Check for missing values in 'Temperature' and 'Humidity' columns
        if combined_data['Temperature'].isnull().sum() > 0 or combined_data['Humidity'].isnull().sum() > 0:
            # Train and fill missing 'Temperature' values if any
            if combined_data['Temperature'].isnull().sum() > 0:
                combined_data = neural_network.train_and_fill(combined_data, 'Temperature', 100)
            # Train and fill missing 'Humidity' values if any
            if combined_data['Humidity'].isnull().sum() > 0:
                combined_data = neural_network.train_and_fill(combined_data, 'Humidity', 100)
        else:
            print('No null values detected!!!!!!')

        return combined_data
