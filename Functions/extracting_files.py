import pandas as pd
import glob
import os
import zipfile

class ExtractingFiles:
    
    @staticmethod
    def extract_zip(file_path, extract_to):
        os.makedirs(extract_to, exist_ok=True)

        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)

        return f'Files extracted to {extract_to}'

    @staticmethod
    def extract_data_from_sheet(df):
        sensor_name = df.iloc[0, 1]

        extracted_df = df.iloc[3:].reset_index(drop=True)

        extracted_df.columns = ['Time', 'Temperature', 'Humidity']
        extracted_df['Time'] = pd.to_datetime(extracted_df['Time'], format='%d-%m-%Y %H:%M')
        
        extracted_df['Sensor Info'] = sensor_name

        return extracted_df

    @staticmethod
    def split_sensor_info(sensor_info):
        parts = sensor_info.split(' ')
        return pd.Series([parts[0], sensor_info])
    
    @staticmethod
    def read_all_excel_files(directory):
        excel_files = glob.glob(os.path.join(directory, '**', '*.xlsx*'), recursive=True)
        data_frames = []

        for file in excel_files:
            print(f'Reading file: {file}')
            # Read each sheet in the Excel file
            xls = pd.ExcelFile(file)
            for sheet_name in xls.sheet_names:
                df = pd.read_excel(file, sheet_name=sheet_name)
                extracted_df = ExtractingFiles.extract_data_from_sheet(df)
                data_frames.append(extracted_df)

        # Combine all DataFrames
        combined_data = pd.concat(data_frames, ignore_index=True)

        # Apply the split_sensor_info function
        combined_data[['Sensor ID', 'Sensor Name']] = combined_data['Sensor Info'].apply(ExtractingFiles.split_sensor_info)

        return combined_data
    
    @staticmethod
    def sorting_data(data, keys_list):
        data['Sensor ID'] = pd.Categorical(data['Sensor ID'], categories=keys_list, ordered=True)
        df_sorted = data.sort_values(by='Sensor ID')
        df_sorted = df_sorted[~pd.isna(df_sorted['Sensor ID'])]
        df_sorted['Sensor ID'] = df_sorted['Sensor ID'].astype(str)
        df_sorted.reset_index(drop=True, inplace=True)

        return df_sorted
