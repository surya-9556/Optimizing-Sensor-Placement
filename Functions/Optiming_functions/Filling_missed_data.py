import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization, Activation
from tensorflow.keras.regularizers import l1
from tensorflow.keras.optimizers import Nadam

class FillingMissedData():
    
    @staticmethod
    def standerdizing_data(last_day_records,column):
        x = []

        for i in range(1, 201):
        # Example data preparation
            temp_storage = last_day_records.groupby('Sensor ID',observed=False).nth(i - 1).reset_index()

            required_column = temp_storage[column].values

            x.append(required_column)

        percentage = 0.8
        num_elements = int(len(x) * percentage)

        X = x[:num_elements]
        y = x[:num_elements]

        x = np.array(X)
        y = np.array(y)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(x)

        return X_scaled, y
    
    @staticmethod
    def neural_network_model(X_scaled):
        # Initialize the model
        model = Sequential()

        # Input layer
        model.add(Input(shape=(X_scaled.shape[1],)))

        # First Dense layer with BatchNormalization, Activation, and Dropout
        model.add(Dense(X_scaled.shape[1], kernel_regularizer=l1(0.01)))
        model.add(BatchNormalization())  # Batch normalization
        model.add(Activation('relu'))
        model.add(Dropout(0.5))  # Dropout layer

        # Add 50 Dense layers with BatchNormalization, Activation, and Dropout
        for _ in range(50):
            model.add(Dense(X_scaled.shape[1], kernel_regularizer=l1(0.01)))
            model.add(BatchNormalization())  # Batch normalization
            model.add(Activation('relu'))
            model.add(Dropout(0.5))  # Dropout layer

        # Output layer
        model.add(Dense(X_scaled.shape[1]))

        model.compile(optimizer=Nadam(learning_rate=0.001), loss='mse')

        return model

