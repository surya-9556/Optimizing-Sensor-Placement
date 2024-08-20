import numpy as np
import cv2
from scipy.fftpack import fftn, fftshift

class NyquistTheorem:
    
    @staticmethod
    def process_video_fft_average(video_path):
        """
        Processes a video file to compute the average FFT (Fast Fourier Transform) 
        of all frames and the magnitude spectrum of the average FFT.

        Parameters:
        - video_path: Path to the video file.

        Returns:
        - fft_avg: Average FFT of all frames in the video.
        - magnitude_spectrum: Magnitude spectrum of the average FFT.
        """
        cap = cv2.VideoCapture(video_path)
        frames_processed = 0
        fft_sum = None
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert the frame to grayscale
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply FFT to the grayscale frame using scipy.fftpack.fftn
            f_transform = fftn(gray_frame)
            f_transform_shifted = fftshift(f_transform)
            
            if fft_sum is None:
                fft_sum = np.zeros_like(f_transform_shifted, dtype=np.complex128)
            
            # Accumulate the FFT results
            fft_sum += f_transform_shifted
            frames_processed += 1
        
        cap.release()
        
        # Calculate the average FFT across all frames
        fft_avg = fft_sum / frames_processed
        
        # Compute the magnitude spectrum of the average FFT
        magnitude_spectrum = 20 * np.log(np.abs(fft_avg))
        
        return fft_avg, magnitude_spectrum
    
    @staticmethod
    def get_video_info(file_path):
        """
        Retrieves basic information about a video file such as frame count, width, and height.

        Parameters:
        - file_path: Path to the video file.

        Returns:
        - frame_count: Number of frames in the video.
        - width: Width of the video frames.
        - height: Height of the video frames.
        """
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
        """
        Reads a video file and converts each frame to grayscale.

        Parameters:
        - file_path: Path to the video file.

        Returns:
        - video_data: Array of grayscale frames from the video.
        """
        frames = []

        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            raise ValueError("Error opening video file")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert the frame to grayscale
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(gray_frame)

        cap.release()

        video_data = np.array(frames)
        return video_data

    @staticmethod
    def reading_video(time_frames, x_dim, y_dim, video_data):
        """
        Performs 3D FFT (Fast Fourier Transform) on the video data and calculates various
        frequency domain metrics based on the Nyquist theorem.

        Parameters:
        - time_frames: Number of frames in the video.
        - x_dim: Width of each frame.
        - y_dim: Height of each frame.
        - video_data: Array of grayscale frames from the video.

        Returns:
        - nyquist_results: Dictionary containing FFT data, magnitude spectrum, frequencies, and sampling rates.
        """
        # Apply 3D FFT to the video data
        fft_data = fftn(video_data)

        # Calculate the magnitude spectrum of the FFT data
        magnitude_spectrum = np.abs(fft_data)

        # Compute frequency bins for each dimension
        frequencies_time = np.fft.fftfreq(time_frames)
        frequencies_x = np.fft.fftfreq(x_dim)
        frequencies_y = np.fft.fftfreq(y_dim)

        # Calculate Nyquist sampling rates for each dimension
        sampling_rate_time = 2 * max(frequencies_time)
        sampling_rate_x = 2 * max(frequencies_x)
        sampling_rate_y = 2 * max(frequencies_y)

        # Store results in a dictionary
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
