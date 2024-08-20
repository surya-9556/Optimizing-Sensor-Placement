import os
import matplotlib.pyplot as plt
import numpy as np
import keras as ker

class AnomalyDetection:
    
    @staticmethod
    def detect_anomalies(sensor_data, sensor_id, temperature_color, humidity_color):
        """
        Detects anomalies in temperature and humidity data for a specific sensor 
        and generates histograms with highlighted anomalies.

        Parameters:
        - sensor_data: DataFrame containing the sensor data (Temperature and Humidity columns).
        - sensor_id: Identifier for the sensor being analyzed.
        - temperature_color: Color for the temperature histogram.
        - humidity_color: Color for the humidity histogram.

        Returns:
        - sensor_data: DataFrame with added columns indicating anomalies ('Temperature_anomaly' and 'Humidity_anomaly').
        """

        # Create directories for storing anomaly plots if they don't exist
        temperature_folder = os.path.join('anomalies', 'temperature')
        humidity_folder = os.path.join('anomalies', 'humidity')
        os.makedirs(temperature_folder, exist_ok=True)
        os.makedirs(humidity_folder, exist_ok=True)
        
        # Define a helper function for plotting histograms
        def plot_histogram(column, color):
            fig, ax = plt.subplots(figsize=(9, 6))
            
            # Remove NaN values for analysis
            data_without_nans = sensor_data[column].dropna()
            mean = data_without_nans.mean()
            std_dev = data_without_nans.std()
            lower_bound = mean - 2 * std_dev
            upper_bound = mean + 2 * std_dev
            
            # Mark data points outside the 2-sigma range as anomalies
            sensor_data[f'{column}_anomaly'] = np.where(
                (sensor_data[column] < lower_bound) | (sensor_data[column] > upper_bound), 1, 0)
            
            # Plot the histogram of the data
            ax.hist(data_without_nans, bins=30, alpha=0.6, color=color, density=True)
            
            # Plot the normal distribution curve
            x = np.linspace(mean - 4 * std_dev, mean + 4 * std_dev, 1000)
            p = (1 / (std_dev * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std_dev) ** 2)
            ax.plot(x, p, 'k', linewidth=2)
            
            # Highlight the anomalies in the plot
            anomalies = sensor_data[(sensor_data[f'{column}_anomaly'] == 1) & (~sensor_data[column].isnull())]
            ax.scatter(anomalies[column], np.zeros(len(anomalies)), color='red')
            
            # Set plot labels and title
            ax.set_title(f'Normal Distribution of {column} for Sensor {sensor_id}')
            ax.set_xlabel(column)
            ax.set_ylabel('Density')
            ax.axvline(x=lower_bound, color='blue', linestyle='--', label='Lower 2-Sigma')
            ax.axvline(x=upper_bound, color='blue', linestyle='--', label='Upper 2-Sigma')
            ax.legend()
            
            # Save the plot as a PNG file
            fig_path = os.path.join(temperature_folder if column == 'Temperature' else humidity_folder,
                                    f'sensor_{sensor_id}_{column}_anomalies.png')
            plt.tight_layout()
            plt.savefig(fig_path)
            plt.close(fig)
        
        # Plot and save the histogram for temperature data
        plot_histogram('Temperature', temperature_color)
        
        # Plot and save the histogram for humidity data
        plot_histogram('Humidity', humidity_color)
        
        return sensor_data
    
    @staticmethod
    def model(input_shape):
        """
        Builds and compiles a neural network model for anomaly detection.

        Parameters:
        - input_shape: Integer representing the shape of the input data.

        Returns:
        - Compiled Keras model.
        """
        model = ker.models.Sequential([
            ker.layers.Input(shape=(input_shape,)),
            ker.layers.Dense(128, activation='relu'),
            ker.layers.Dense(64, activation='relu'),
            ker.layers.Dense(64, activation='relu'),
            ker.layers.Dense(32, activation='relu'),
            ker.layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model
    
    @staticmethod
    def training_anomaly_model(x_train_scaled, y_train_temp, y_train_humidity, target):
        """
        Trains a neural network model on the provided training data to detect anomalies.

        Parameters:
        - x_train_scaled: Scaled features for training.
        - y_train_temp: Target temperature data for training.
        - y_train_humidity: Target humidity data for training.
        - target: String specifying whether the target is 'temperature' or 'humidity'.

        Returns:
        - Trained Keras model.
        """
        # Select the appropriate target data based on the specified target
        if target == 'temperature':
            y_train = y_train_temp
        elif target == 'humidity':
            y_train = y_train_humidity
        else:
            raise ValueError("Target must be 'temperature' or 'humidity'")

        # Build and train the model
        model = AnomalyDetection.model(x_train_scaled.shape[1])
        model.fit(x_train_scaled, y_train, epochs=20, batch_size=64, validation_split=0.3)
        return model
