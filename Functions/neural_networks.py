import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
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
        # model.fit(X, y, epochs=20, batch_size=32, validation_split=0.3)
        # Fit model and store history
        history = model.fit(X, y, epochs=20, batch_size=32, validation_split=0.3)

        # Normalize the losses
        train_loss = np.array(history.history['loss'])
        # val_loss = np.array(history.history['val_loss'])
        normalized_train_loss = train_loss / train_loss[0]
        # normalized_val_loss = val_loss / val_loss[0]

        # Plot the losses
        plt.figure(figsize=(10, 6))
        plt.plot(normalized_train_loss, label='Training Loss')
        # plt.plot(normalized_val_loss, label='Validation Loss')
        plt.title(f'Normalized MSE Loss over Epochs for {feature_name}')
        plt.xlabel('Epochs')
        plt.ylabel('Normalized MSE Loss')
        plt.legend()
        plt.grid(True)
        plt.show()
        filled_data = self.fill_missing_values(data, model, feature_name, n_steps)
        return filled_data
