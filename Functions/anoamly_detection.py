import os
import matplotlib.pyplot as plt
import numpy as np

class AnomalyDetection:
    
    @staticmethod
    def detect_anomalies(sensor_data, sensor_id, temperature_color, humidity_color):
        # Create directories if they don't exist
        temperature_folder = os.path.join('anomalies', 'temperature')
        humidity_folder = os.path.join('anomalies', 'humidity')
        os.makedirs(temperature_folder, exist_ok=True)
        os.makedirs(humidity_folder, exist_ok=True)
        
        # Define a plotting function for reuse
        def plot_histogram(column, color):
            fig, ax = plt.subplots(figsize=(9, 6))
            data_without_nans = sensor_data[column].dropna()
            mean = data_without_nans.mean()
            std_dev = data_without_nans.std()
            lower_bound = mean - 2 * std_dev
            upper_bound = mean + 2 * std_dev
            
            # Detect anomalies (excluding NaN values)
            sensor_data[f'{column}_anomaly'] = np.where(
                (sensor_data[column] < lower_bound) | (sensor_data[column] > upper_bound), 1, 0)
            
            # Plot histogram
            ax.hist(data_without_nans, bins=30, alpha=0.6, color=color, density=True)
            
            # Plot normal distribution curve
            x = np.linspace(mean - 4 * std_dev, mean + 4 * std_dev, 1000)
            p = (1 / (std_dev * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std_dev) ** 2)
            ax.plot(x, p, 'k', linewidth=2)
            
            # Highlight anomalies
            anomalies = sensor_data[(sensor_data[f'{column}_anomaly'] == 1) & (~sensor_data[column].isnull())]
            ax.scatter(anomalies[column], np.zeros(len(anomalies)), color='red')
            
            # Set plot labels and title
            ax.set_title(f'Normal Distribution of {column} for Sensor {sensor_id}')
            ax.set_xlabel(column)
            ax.set_ylabel('Density')
            ax.axvline(x=lower_bound, color='blue', linestyle='--', label='Lower 2-Sigma')
            ax.axvline(x=upper_bound, color='blue', linestyle='--', label='Upper 2-Sigma')
            ax.legend()
            
            # Save plot
            fig_path = os.path.join(temperature_folder if column == 'Temperature' else humidity_folder,
                                    f'sensor_{sensor_id}_{column}_anomalies.png')
            plt.tight_layout()
            plt.savefig(fig_path)
            plt.close(fig)
        
        # Plot temperature histogram
        plot_histogram('Temperature', temperature_color)
        
        # Plot humidity histogram
        plot_histogram('Humidity', humidity_color)
        
        return sensor_data