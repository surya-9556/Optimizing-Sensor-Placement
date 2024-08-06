import numpy as np
import pandas as pd
import os
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization, Activation
from tensorflow.keras.regularizers import l1
from tensorflow.keras.optimizers import Nadam

class AlternativeMethod():
    
    @staticmethod
    def standerdizing_data(latest_month_data, latest_month_data_1, column):
        x = []
        y = []

        for i in range(1, 5001):
            temp_storage = latest_month_data.groupby('Sensor ID', observed=False).nth(i - 1).reset_index()
            x.append(temp_storage[column].values)

            temp_storage_1 = latest_month_data_1.groupby('Sensor ID', observed=False).nth(i - 1).reset_index()
            y.append(temp_storage_1[column].values)

        x = np.array(x)
        y = np.array(y)

        return x, y
    
    @staticmethod
    def shuffle_data(data):# Reshape the array to 1D for shuffling
        reshaped_data = data.reshape(-1)

        # Shuffle the array
        np.random.shuffle(reshaped_data)

        # Reshape the shuffled array back to its original shape
        shuffled_data = reshaped_data.reshape(data.shape)

        # # Print or use shuffled_data as needed
        # print(shuffled_data)

        return shuffled_data
    
    @staticmethod
    def neural_network_model(x, y):
        model = Sequential()

        model.add(Input(shape=(x.shape[1],)))

        model.add(Dense(x.shape[1],kernel_regularizer='l2'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        # model.add(Dropout(0.3))

        for _ in range(10):
            model.add(Dense(y.shape[1],kernel_regularizer='l2'))
            model.add(BatchNormalization())
            model.add(Activation('relu'))
            # model.add(Dropout(0.5))

        for _ in range(5):
            model.add(Dense(2 * y.shape[1],kernel_regularizer='l2'))
            model.add(BatchNormalization())
            model.add(Activation('relu'))
            # model.add(Dropout(0.7))

        for _ in range(4):
            model.add(Dense(4 * y.shape[1],kernel_regularizer='l2'))
            model.add(BatchNormalization())
            model.add(Activation('relu'))
            # model.add(Dropout(0.7))

        model.add(Dense(8 * y.shape[1],kernel_regularizer='l2'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        # model.add(Dropout(0.9))

        model.add(Dense(y.shape[1]))
        model.add(Activation('relu'))

        model.compile(optimizer=Nadam(learning_rate=0.0001), loss='mse')

        return model
    
    @staticmethod
    def predict_future_temperatures(model, reduced_df_sorted, reduced_df_sorted_1, selected_sensors_list, column, num_steps):
        # Initial prediction
        predicting_data = reduced_df_sorted.groupby('Sensor ID', observed=False).nth(-1)[['Sensor ID', column]]
        temperature_input = np.array(predicting_data[column].tolist()).reshape(1, -1)
        predictions = model.predict(temperature_input)

        # Create DataFrame for the first prediction
        predicting_next_data = pd.DataFrame({
            "Sensor ID": reduced_df_sorted_1['Sensor ID'].unique(),
            "Predicted_"+column: predictions[0],
            'Actual_'+column: reduced_df_sorted_1.groupby('Sensor ID', observed=False).nth(-1)[column].tolist()
            # 'Actual_'+column:y_temp_val[0]
        })

        all_predictions = []

        all_predictions.append(predicting_next_data)
        
        for step in range(1, num_steps):
            predicting_data = predicting_next_data[predicting_next_data['Sensor ID'].isin(selected_sensors_list)]
            temperature_input = np.array(predicting_data['Predicted_'+column].tolist()).reshape(1, -1)
            
            predictions = model.predict(temperature_input)

            predicting_next_data = pd.DataFrame({
                "Sensor ID": reduced_df_sorted_1['Sensor ID'].unique(),
                "Predicted_"+column: predictions[0],
                'Actual_'+column: np.round(predicting_next_data['Predicted_'+column].tolist(),2)
                # 'Actual_'+column:y_temp_val[step]
            })
            
            # Store the current step predictions
            all_predictions.append(predicting_next_data)

        return all_predictions
    
    
    @staticmethod
    def error_plotting(output_dir,predictions_over_time,column, annotate):

        frames = []

        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)

        os.makedirs(output_dir)

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