import numpy as np
import cv2
from scipy.fftpack import fftn

class NyquistTheorem():
    @staticmethod
    def get_video_info(file_path):
        video = cv2.VideoCapture(file_path)

        if not video.isOpened():
            raise ValueError("Error opening video file")

        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

        video.release()

        return frame_count, width, height
    
    @staticmethod
    def read_video_file(file_path):
        frames = []

        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            raise ValueError("Error opening video file")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(gray_frame)

        cap.release()

        video_data = np.array(frames)
        return video_data

    @staticmethod
    def reading_video(time_frames,x_dim,y_dim,video_data):

        fft_data = fftn(video_data)

        magnitude_spectrum = np.abs(fft_data)

        frequencies_time = np.fft.fftfreq(time_frames)
        frequencies_x = np.fft.fftfreq(x_dim)
        frequencies_y = np.fft.fftfreq(y_dim)

        sampling_rate_time = 2 * max(frequencies_time)
        sampling_rate_x = 2 * max(frequencies_x)
        sampling_rate_y = 2 * max(frequencies_y)

        nyquist_results = {
            "fft_data": fft_data,
            "magnitude_spectrum": magnitude_spectrum,
            "frequencies_time": frequencies_time,
            "frequencies_x": frequencies_x,
            "frequencies_y": frequencies_y,
            "sampling_rate_time": sampling_rate_time,
            "sampling_rate_x": sampling_rate_x,
            "sampling_rate_y": sampling_rate_y,
        }

        return nyquist_results