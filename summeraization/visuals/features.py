import cv2
import os
import pandas as pd
import numpy as np


def is_frame_acceptable(img_path: str) -> bool:
    """Apply rule-based filters to determine if a frame is worth keeping."""
    img = cv2.imread(img_path)
    if img is None:
        return False

    height, width, _ = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    contrast = float(np.std(gray))
    brightness = float(np.mean(gray))
    dark_ratio = float(np.sum(gray < 30) / (height * width))
    bright_ratio = float(np.sum(gray > 220) / (height * width))
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    edge_density = float(np.sum(cv2.Canny(gray, 50, 150) > 0) / (height * width))
    hist_var = float(np.var(cv2.calcHist([gray], [0], None, [256], [0, 256])))
    unique_colors = int(len(np.unique(gray)))
    color_diversity = float(unique_colors / 256.0)

    if brightness < 20: return False
    if contrast < 10: return False
    if dark_ratio > 0.8 or bright_ratio > 0.9: return False
    if laplacian_var < 100: return False
    if edge_density < 0.01: return False
    if color_diversity < 0.1 or unique_colors < 20: return False
    if hist_var < 100: return False

    return True


def evaluate_feature_quality(
    input_csv: str = "tmp/frames/keyframes.csv",
    output_csv: str = "tmp/frames/descriptions.csv"
    ) -> pd.DataFrame:
    """Evaluate feature quality and flag each frame accordingly."""
    df = pd.read_csv(input_csv)
    feature_flags = []

    for idx, row in df.iterrows():
        frame_path = row.get("path") or row.get("keyframe")
        if not frame_path or not os.path.exists(frame_path):
            feature_flags.append("not_important")
            continue

        if is_frame_acceptable(frame_path):
            feature_flags.append("important")
        else:
            feature_flags.append("not_important")

    df["feature_flag"] = feature_flags
    df.to_csv(output_csv, index=False)
    print(f"✅ Feature flags written to {output_csv}")
    return df
