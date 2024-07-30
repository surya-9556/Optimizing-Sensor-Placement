import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from scipy import stats

class ComparingVideos:
    @staticmethod
    def extract_frames(video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        return frames
    
    @staticmethod
    def compare_frames(frame1, frame2):
        frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        score, _ = ssim(frame1_gray, frame2_gray, full=True)
        return score
    
    @classmethod
    def compare_videos(cls, video1_frames, video2_frames):
        if len(video1_frames) != len(video2_frames):
            print("Warning: Videos have different number of frames")
        
        similarity_scores = []
        for f1, f2 in zip(video1_frames, video2_frames):
            score = cls.compare_frames(f1, f2)
            similarity_scores.append(score)
        
        return similarity_scores
    
    @staticmethod
    def overall_similarity(similarity_scores):
        return np.mean(similarity_scores)
    
    @staticmethod
    def perform_statistical_tests(similarity_scores):
        # Perform a one-sample t-test on similarity scores
        t_statistic, p_value_ttest = stats.ttest_1samp(similarity_scores, 0)
        
        # Perform a Wilcoxon signed-rank test
        w_statistic, p_value_wilcoxon = stats.wilcoxon(similarity_scores - np.mean(similarity_scores))
        
        # Perform a Kolmogorov-Smirnov test
        d_statistic, p_value_ks = stats.kstest(similarity_scores, 'norm', args=(np.mean(similarity_scores), np.std(similarity_scores)))
        
        return {
            't-test': (t_statistic, p_value_ttest),
            'Wilcoxon signed-rank test': (w_statistic, p_value_wilcoxon),
            'Kolmogorov-Smirnov test': (d_statistic, p_value_ks)
        }