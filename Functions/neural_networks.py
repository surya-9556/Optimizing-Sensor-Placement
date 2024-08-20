import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import keras as ker

class NeuralNetwork:
    
    @staticmethod
    def create_sequences(data, n_steps):
        """
        Creates sequences of data for training the model. Each sequence will be used as input features (X),
        and the subsequent value in the series as the target (y).

        Parameters:
        - data: Array or list of data points.
        - n_steps: Number of time steps to use for each sequence.

        Returns:
        - X: Array of input sequences.
        - y: Array of target values.
        """
        X, y = [], []
        for i in range(n_steps, len(data)):
            current_data_slice = data[i-n_steps:i]
            # Only append sequences that don't contain NaN values
            if not np.isnan(data[i]) and not np.isnan(current_data_slice).any():
                X.append(current_data_slice)
                y.append(data[i])
        return np.array(X), np.array(y)
    
    @staticmethod
    def build_model(input_shape):
        """
        Builds and compiles a sequential neural network model for time series prediction.

        Parameters:
        - input_shape: Shape of the input data (number of time steps).

        Returns:
        - model: Compiled Keras model ready for training.
        """
        model = ker.models.Sequential([
            ker.layers.Input(shape=input_shape),
            ker.layers.Dense(128, activation='relu'),
            ker.layers.Dense(128, activation='relu'),
            ker.layers.Dense(64, activation='relu'),
            ker.layers.Dense(64, activation='relu'),
            ker.layers.Dense(32, activation='relu'),
            ker.layers.Dense(1)  # Output layer for predicting the next value
        ])
        model.compile(optimizer='adam', loss='mse')  # Compile the model with Adam optimizer and MSE loss
        return model
    
    @staticmethod
    def fill_missing_values(data, model, feature_name, n_steps):
        """
        Fills missing values in the data using the trained neural network model.

        Parameters:
        - data: DataFrame containing the feature with missing values.
        - model: Trained Keras model used to predict missing values.
        - feature_name: Name of the feature/column in the DataFrame with missing values.
        - n_steps: Number of time steps to use for prediction.

        Returns:
        - data_copy: DataFrame with missing values filled in the specified feature.
        """
        data_copy = data.copy()  # Create a copy of the data to avoid modifying the original DataFrame
        feature_data = data_copy[feature_name].values  # Extract the feature data as a NumPy array

        for i in range(n_steps, len(feature_data)):
            if np.isnan(feature_data[i]):  # Check if the current value is NaN
                # Skip if any of the preceding n_steps values are NaN
                if np.isnan(feature_data[i-n_steps:i]).any():
                    continue
                # Prepare the input sequence and predict the missing value
                X_input = feature_data[i-n_steps:i].reshape(1, -1)
                predicted_value = model.predict(X_input, verbose=0)
                feature_data[i] = np.round(predicted_value.item(), 2)  # Fill the missing value with the predicted value

        data_copy[feature_name] = feature_data  # Update the DataFrame with filled values
        return data_copy
    
    def train_and_fill(self, data, feature_name, n_steps):
        """
        Trains a neural network model on the provided data, and uses it to fill missing values.

        Parameters:
        - data: DataFrame containing the feature with missing values.
        - feature_name: Name of the feature/column in the DataFrame with missing values.
        - n_steps: Number of time steps to use for training and prediction.

        Returns:
        - filled_data: DataFrame with missing values filled in the specified feature.
        """
        # Extract the feature data and create sequences for training
        feature_data = data[feature_name].values
        X, y = self.create_sequences(feature_data, n_steps)
        input_shape = (X.shape[1],)  # Determine the shape of the input data
        model = self.build_model(input_shape)  # Build the model

        # Train the model and store the training history
        history = model.fit(X, y, epochs=20, batch_size=32, validation_split=0.3)

        # Normalize the training losses
        train_loss = np.array(history.history['loss'])
        normalized_train_loss = train_loss / train_loss[0]

        # Plot the normalized training losses over epochs
        plt.figure(figsize=(10, 6))
        plt.plot(normalized_train_loss, label='Training Loss')
        plt.title(f'Normalized MSE Loss over Epochs for {feature_name}')
        plt.xlabel('Epochs')
        plt.ylabel('Normalized MSE Loss')
        plt.legend()
        plt.grid(True)
        plt.show()

        # Fill missing values using the trained model
        filled_data = self.fill_missing_values(data, model, feature_name, n_steps)
        return filled_data
