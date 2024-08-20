import numpy as np
import pandas as pd
import os
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization, Activation
from tensorflow.keras.optimizers import Nadam

class AlternativeMethod:
    
    @staticmethod
    def standerdizing_data(latest_month_data, latest_month_data_1, column):
        """
        Standardizes the data from two different months by extracting values for a specific column
        across all sensors, grouping by 'Sensor ID', and storing the results in arrays.

        Parameters:
        - latest_month_data: DataFrame containing data from the most recent month.
        - latest_month_data_1: DataFrame containing data from the previous month.
        - column: The column name to be standardized.

        Returns:
        - x: Array of standardized data for the latest month.
        - y: Array of standardized data for the previous month.
        """
        x = []
        y = []

        for i in range(1, 5001):
            # Extract data for the i-th occurrence of each Sensor ID
            temp_storage = latest_month_data.groupby('Sensor ID', observed=False).nth(i - 1).reset_index()
            x.append(temp_storage[column].values)

            temp_storage_1 = latest_month_data_1.groupby('Sensor ID', observed=False).nth(i - 1).reset_index()
            y.append(temp_storage_1[column].values)

        x = np.array(x)
        y = np.array(y)

        return x, y
    
    @staticmethod
    def shuffle_data(data):
        """
        Shuffles the given data while preserving its shape.

        Parameters:
        - data: The data array to be shuffled.

        Returns:
        - shuffled_data: The shuffled data array with the same shape as the original.
        """
        # Reshape the array to 1D for shuffling
        reshaped_data = data.reshape(-1)

        # Shuffle the array
        np.random.shuffle(reshaped_data)

        # Reshape the shuffled array back to its original shape
        shuffled_data = reshaped_data.reshape(data.shape)

        return shuffled_data
    
    @staticmethod
    def neural_network_model(x, y):
        """
        Builds and compiles a deep neural network model for data prediction.

        Parameters:
        - x: Input data array (features).
        - y: Target data array (labels).

        Returns:
        - model: Compiled Keras Sequential model.
        """
        model = Sequential()

        # Input layer
        model.add(Input(shape=(x.shape[1],)))

        # First layer matching input shape
        model.add(Dense(x.shape[1], kernel_regularizer='l2'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        # Add layers in stages with increasing complexity
        for _ in range(10):
            model.add(Dense(y.shape[1], kernel_regularizer='l2'))
            model.add(BatchNormalization())
            model.add(Activation('relu'))

        for _ in range(5):
            model.add(Dense(2 * y.shape[1], kernel_regularizer='l2'))
            model.add(BatchNormalization())
            model.add(Activation('relu'))

        for _ in range(4):
            model.add(Dense(4 * y.shape[1], kernel_regularizer='l2'))
            model.add(BatchNormalization())
            model.add(Activation('relu'))

        # Final layer with increased complexity
        model.add(Dense(8 * y.shape[1], kernel_regularizer='l2'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        # Output layer
        model.add(Dense(y.shape[1]))
        model.add(Activation('relu'))

        # Compile the model with Nadam optimizer and mean squared error loss
        model.compile(optimizer=Nadam(learning_rate=0.0001), loss='mse')

        return model
    
    @staticmethod
    def predict_future_temperatures(model, reduced_df_sorted, reduced_df_sorted_1, selected_sensors_list, column, num_steps):
        """
        Predicts future temperature values for a given number of steps.

        Parameters:
        - model: Trained neural network model.
        - reduced_df_sorted: DataFrame with sorted sensor data for prediction.
        - reduced_df_sorted_1: DataFrame with sorted sensor data for comparison.
        - selected_sensors_list: List of sensors to predict for.
        - column: The column name representing the temperature data.
        - num_steps: The number of future steps to predict.

        Returns:
        - all_predictions: List of DataFrames containing predictions for each step.
        """
        # Initial prediction based on the most recent data
        predicting_data = reduced_df_sorted.groupby('Sensor ID', observed=False).nth(-1)[['Sensor ID', column]]
        temperature_input = np.array(predicting_data[column].tolist()).reshape(1, -1)
        predictions = model.predict(temperature_input)

        # Create DataFrame for the first prediction
        predicting_next_data = pd.DataFrame({
            "Sensor ID": reduced_df_sorted_1['Sensor ID'].unique(),
            "Predicted_" + column: predictions[0],
            'Actual_' + column: reduced_df_sorted_1.groupby('Sensor ID', observed=False).nth(-1)[column].tolist()
        })

        all_predictions = []
        all_predictions.append(predicting_next_data)
        
        # Generate predictions for the specified number of steps
        for step in range(1, num_steps):
            predicting_data = predicting_next_data[predicting_next_data['Sensor ID'].isin(selected_sensors_list)]
            temperature_input = np.array(predicting_data['Predicted_' + column].tolist()).reshape(1, -1)
            predictions = model.predict(temperature_input)

            predicting_next_data = pd.DataFrame({
                "Sensor ID": reduced_df_sorted_1['Sensor ID'].unique(),
                "Predicted_" + column: predictions[0],
                'Actual_' + column: np.round(predicting_next_data['Predicted_' + column].tolist(), 2)
            })
            
            # Store the current step predictions
            all_predictions.append(predicting_next_data)

        return all_predictions
    
    @staticmethod
    def error_plotting(output_dir, predictions_over_time, column, annotate):
        """
        Plots and saves heatmaps of prediction errors over time.

        Parameters:
        - output_dir: Directory where the heatmaps will be saved.
        - predictions_over_time: DataFrame with predictions over time.
        - column: The column name representing the temperature data.
        - annotate: Boolean flag to annotate the heatmap cells.

        Returns:
        - frames: List of file paths to the saved heatmap images.
        """
        frames = []

        # Remove existing directory and create a new one
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)

        # Generate heatmaps for a specified number of time steps
        for i in range(2, 102):
            temp_predict_data = predictions_over_time.groupby('Sensor ID', observed=False).nth(i - 1).reset_index()
            reconstructed_matrix = temp_predict_data[column].values.reshape(8, 7)

            plt.figure(figsize=(10, 6))
            sns.heatmap(reconstructed_matrix, annot=annotate, fmt=".2f", cmap='YlGnBu', cbar=False, xticklabels=False, yticklabels=False)
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

            output_file = os.path.join(output_dir, f"error_temp_heatmap_{i}.png")
            plt.savefig(output_file)
            plt.close()

            frames.append(output_file)

        return frames
