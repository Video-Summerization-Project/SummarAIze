import pandas as pd
import os
from summeraization.visuals.features import evaluate_feature_quality
from summeraization.visuals.evaluator import evaluate_llm_importance
from summeraization.visuals.describer import describe_frame


def run_visual_pipeline(
    keyframes_csv: str = "tmp/frames/keyframes.csv",
    output_csv: str = "tmp/frames/descriptions.csv"
) -> None:
    """Run the full visual processing pipeline with flag-based logic."""

    print("🔍 Step 1: Evaluating visual features...")
    evaluate_feature_quality(keyframes_csv, output_csv)

    print("🧠 Step 2: Evaluating semantic importance (LLM)...")
    evaluate_llm_importance(output_csv, output_csv)

    print("📋 Step 3: Generating visual descriptions...")
    add_descriptions_to_csv(output_csv)

def add_descriptions_to_csv(csv_path: str = "tmp/frames/descriptions.csv") -> None:
    if not os.path.exists(csv_path):
        print(f"❌ CSV not found at: {csv_path}")
        return

    df = pd.read_csv(csv_path)

    explanations = []
    summaries = []

    for idx, row in df.iterrows():
        if (
            row.get("feature_flag") != "important" or
            row.get("llm_flag") != "important"
        ):
            explanations.append("Skipped")
            summaries.append("Skipped")
            continue

        image_path = row.get("path") or row.get("keyframe") or row.get("keyframes")
        if not image_path or not os.path.exists(image_path):
            explanations.append("Missing")
            summaries.append("Missing")
            continue

        try:
            result = describe_frame(image_path)
            explanations.append(result["explanation"])
            summaries.append(result["summary"])
            print(f"📝 Described: {os.path.basename(image_path)}")
        except Exception as e:
            print(f"⚠️ Error describing {image_path}: {e}")
            explanations.append("Error")
            summaries.append("Error")

    df["explanation"] = explanations
    df["summary"] = summaries
    df.to_csv(csv_path, index=False)
    print(f"✅ Descriptions saved to {csv_path}")
