import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from scipy import stats

class ComparingVideos:
    
    @staticmethod
    def extract_frames(video_path):
        """
        Extracts frames from a video file.

        Parameters:
        - video_path: Path to the video file.

        Returns:
        - frames: A list of frames extracted from the video.
        """
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break  # Exit loop if there are no more frames
            frames.append(frame)
        cap.release()  # Release the video capture object
        return frames
    
    @staticmethod
    def compare_frames(frame1, frame2):
        """
        Compares two video frames using Structural Similarity Index (SSIM).

        Parameters:
        - frame1: The first frame to compare.
        - frame2: The second frame to compare.

        Returns:
        - score: SSIM score indicating the similarity between the two frames.
        """
        # Convert the frames to grayscale
        frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # Calculate SSIM between the two frames
        score, _ = ssim(frame1_gray, frame2_gray, full=True)
        return score
    
    @classmethod
    def compare_videos(cls, video1_frames, video2_frames):
        """
        Compares frames of two videos frame by frame.

        Parameters:
        - video1_frames: A list of frames from the first video.
        - video2_frames: A list of frames from the second video.

        Returns:
        - similarity_scores: A list of SSIM scores for each frame comparison.
        """
        if len(video1_frames) != len(video2_frames):
            print("Warning: Videos have different number of frames")
        
        similarity_scores = []
        for f1, f2 in zip(video1_frames, video2_frames):
            # Compare each pair of frames and store the SSIM score
            score = cls.compare_frames(f1, f2)
            similarity_scores.append(score)
        
        return similarity_scores
    
    @staticmethod
    def overall_similarity(similarity_scores):
        """
        Calculates the overall similarity between two videos.

        Parameters:
        - similarity_scores: A list of SSIM scores for all frames.

        Returns:
        - mean_similarity: The mean SSIM score indicating overall similarity.
        """
        return np.mean(similarity_scores)
    
    @staticmethod
    def perform_statistical_tests(similarity_scores):
        """
        Performs statistical tests on the similarity scores.

        Parameters:
        - similarity_scores: A list of SSIM scores for all frames.

        Returns:
        - test_results: A dictionary containing results from the t-test, Wilcoxon signed-rank test, 
                        and Kolmogorov-Smirnov test.
        """
        # Perform a one-sample t-test to determine if the mean of the similarity scores is significantly different from zero
        t_statistic, p_value_ttest = stats.ttest_1samp(similarity_scores, 0)
        
        # Perform a Wilcoxon signed-rank test to assess the median similarity score
        w_statistic, p_value_wilcoxon = stats.wilcoxon(similarity_scores - np.mean(similarity_scores))
        
        # Perform a Kolmogorov-Smirnov test to compare the distribution of similarity scores to a normal distribution
        d_statistic, p_value_ks = stats.kstest(similarity_scores, 'norm', 
                                               args=(np.mean(similarity_scores), np.std(similarity_scores)))
        
        return {
            't-test': (t_statistic, p_value_ttest),
            'Wilcoxon signed-rank test': (w_statistic, p_value_wilcoxon),
            'Kolmogorov-Smirnov test': (d_statistic, p_value_ks)
        }
