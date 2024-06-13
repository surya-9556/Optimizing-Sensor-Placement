import numpy as np
import pandas as pd
import tensorflow as tf
import keras as ker

class NeuralNetwork:
    
    @staticmethod
    def create_sequences(data, n_steps):
        X, y = [], []
        for i in range(n_steps, len(data)):
            current_data_slice = data[i-n_steps:i]
            if not np.isnan(data[i]) and not np.isnan(current_data_slice).any():
                X.append(current_data_slice)
                y.append(data[i])
        return np.array(X), np.array(y)
    
    @staticmethod
    def build_model(input_shape):
        model = ker.models.Sequential([
            ker.layers.Input(shape=input_shape),
            ker.layers.Dense(128, activation='relu'),
            ker.layers.Dense(128, activation='relu'),
            ker.layers.Dense(64, activation='relu'),
            ker.layers.Dense(64, activation='relu'),
            ker.layers.Dense(32, activation='relu'),
            ker.layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model
    
    @staticmethod
    def fill_missing_values(data, model, feature_name, n_steps):
        data_copy = data.copy()
        feature_data = data_copy[feature_name].values
        for i in range(n_steps, len(feature_data)):
            if np.isnan(feature_data[i]):
                if np.isnan(feature_data[i-n_steps:i]).any():
                    continue
                X_input = feature_data[i-n_steps:i].reshape(1, -1)
                predicted_value = model.predict(X_input, verbose=0)
                feature_data[i] = np.round(predicted_value.item(), 2)
        data_copy[feature_name] = feature_data
        return data_copy
    
    def train_and_fill(self, data, feature_name, n_steps):
        feature_data = data[feature_name].values
        X, y = self.create_sequences(feature_data, n_steps)
        input_shape = (X.shape[1],)
        model = self.build_model(input_shape)
        model.fit(X, y, epochs=20, batch_size=32, validation_split=0.3)
        filled_data = self.fill_missing_values(data, model, feature_name, n_steps)
        return filled_data
