import warnings
warnings.filterwarnings("ignore")
import os 
import shutil
import json

import time
from .KeyFrameSelection.FeatureExtraction import process_video, save_records
from .KeyFrameSelection.Similarties import hash_filter, clip_filter

def get_keyframes(video_path, model, processor):
    records, fps = process_video(video_path, interval_sec=10)

    min_frames = 10
    max_iterations = 20
    iteration = 0

    hash_threshold = 5
    ssim_threshold = 0.95
    clip_threshold = 0.90

    filtered = records

    while len(filtered) >= min_frames and iteration < max_iterations:
        filtered = hash_filter(
            filtered,
            hash_threshold=hash_threshold,
            ssim_threshold=ssim_threshold,
            ssim_compare_window=5
        )

        filtered = clip_filter(
            filtered,
            model,
            processor,
            similarity_threshold=clip_threshold,
            compare_window=5
        )

        # Threshold tuning
        hash_threshold = max(1, hash_threshold - 1)
        ssim_threshold = max(0.5, ssim_threshold - 0.05)
        clip_threshold = min(0.99, clip_threshold + 0.03)

        iteration += 1
        #print(f"Iter {iteration}: {len(filtered)} frames")

    # Step 3: Save filtered keyframes
    save_records(filtered, fps)
    #print("Keyframe selection process completed successfully.")
    return True

if __name__ == "__main__":
    start = time.time()
    keyframe_dir = 'outputs/keyframes'
    csv_path = 'outputs/keyframes.csv'
    video_path = 'RawVideos\Filters - Mohammad Ayed (720p, h264).mp4'  # Adjust as needed

    if os.path.exists("outputs"):
        shutil.rmtree("outputs")
    os.makedirs("outputs/final_output", exist_ok=True)

    get_keyframes(video_path)
    end = time.time()
    print(f"\nTotal time: {end - start:.2f} sec")
