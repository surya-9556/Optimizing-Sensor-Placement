import pandas as pd
import glob
import os
import zipfile

class ExtractingFiles:
    
    @staticmethod
    def extract_zip(file_path, extract_to):
        """
        Extracts all files from a zip archive to the specified directory.

        Parameters:
        - file_path: Path to the zip file.
        - extract_to: Directory where the contents will be extracted.

        Returns:
        - A string message indicating the extraction location.
        """
        os.makedirs(extract_to, exist_ok=True)  # Ensure the output directory exists

        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)  # Extract all files in the zip archive to the specified directory

        return f'Files extracted to {extract_to}'

    @staticmethod
    def extract_data_from_sheet(df):
        """
        Extracts and formats data from a specific sheet in the Excel file.

        Parameters:
        - df: DataFrame representing the sheet's content.

        Returns:
        - A formatted DataFrame containing time, temperature, humidity, and sensor information.
        """
        sensor_name = df.iloc[0, 1]  # The sensor name is expected to be in the first row, second column

        extracted_df = df.iloc[3:].reset_index(drop=True)  # Skip the first 3 rows and reset the index

        # Rename the columns to more meaningful names
        extracted_df.columns = ['Time', 'Temperature', 'Humidity']
        extracted_df['Time'] = pd.to_datetime(extracted_df['Time'], format='%d-%m-%Y %H:%M')  # Convert 'Time' to datetime format
        
        extracted_df['Sensor Info'] = sensor_name  # Add the sensor name as a new column

        return extracted_df

    @staticmethod
    def split_sensor_info(sensor_info):
        """
        Splits the sensor information into sensor ID and sensor name.

        Parameters:
        - sensor_info: String containing sensor information.

        Returns:
        - A Series with sensor ID and sensor name.
        """
        parts = sensor_info.split(' ')  # Split the string based on spaces
        return pd.Series([parts[0], sensor_info])  # Return the first part as sensor ID and the entire string as sensor name
    
    @staticmethod
    def read_all_excel_files(file):
        """
        Reads all Excel files from a directory or a single Excel file and extracts relevant data.

        Parameters:
        - file: Path to the directory or Excel file.

        Returns:
        - A combined DataFrame with data from all sheets in all Excel files.
        """
        data_frames = []

        if os.path.isdir(file):
            # Get a list of all Excel files in the directory (including subdirectories)
            excel_files = glob.glob(os.path.join(file, '**', '*.xlsx*'), recursive=True)

            for excel_file in excel_files:
                print('excel_file',excel_file)
                xls = pd.ExcelFile(excel_file)  # Load the Excel file
                for sheet_name in xls.sheet_names:
                    df = pd.read_excel(xls, sheet_name=sheet_name)  # Read each sheet
                    extracted_df = ExtractingFiles.extract_data_from_sheet(df)  # Extract relevant data
                    data_frames.append(extracted_df)  # Add to list of data frames
        else:
            xls = pd.ExcelFile(file)  # Load the single Excel file
            for sheet_name in xls.sheet_names:
                df = pd.read_excel(xls, sheet_name=sheet_name)
                extracted_df = ExtractingFiles.extract_data_from_sheet(df)
                data_frames.append(extracted_df)

        # Combine all the extracted data into a single DataFrame
        combined_data = pd.concat(data_frames, ignore_index=True)

        # Split the 'Sensor Info' column into 'Sensor ID' and 'Sensor Name' columns
        combined_data[['Sensor ID', 'Sensor Name']] = combined_data['Sensor Info'].apply(ExtractingFiles.split_sensor_info)
        combined_data = combined_data.drop('Sensor Info', axis=1)  # Drop the original 'Sensor Info' column

        return combined_data
    
    @staticmethod
    def sorting_data(data, keys_list):
        """
        Sorts the data based on the 'Sensor ID' column according to a predefined order.

        Parameters:
        - data: DataFrame containing the data to be sorted.
        - keys_list: List of sensor IDs defining the desired sort order.

        Returns:
        - A sorted DataFrame based on 'Sensor ID'.
        """
        # Ensure 'Sensor ID' is treated as a categorical variable with a specific order
        data['Sensor ID'] = pd.Categorical(data['Sensor ID'], categories=keys_list, ordered=True)
        
        # Sort the DataFrame by 'Sensor ID'
        df_sorted = data.sort_values(by='Sensor ID')
        
        # Remove any rows where 'Sensor ID' is NaN
        df_sorted = df_sorted[~pd.isna(df_sorted['Sensor ID'])]
        
        # Convert 'Sensor ID' back to string type and reset the index
        df_sorted['Sensor ID'] = df_sorted['Sensor ID'].astype(str)
        df_sorted.reset_index(drop=True, inplace=True)

        return df_sorted
