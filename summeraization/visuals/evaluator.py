import re
import json
import os
import pandas as pd
from llm.groq_model import groq_client, MODEL_NAME
from summeraization.visuals.encoder import encode_image_to_base64


def evaluate_llm_importance(
    input_csv: str = "tmp/frames/descriptions.csv",
    output_csv: str = "tmp/frames/descriptions.csv"
) -> pd.DataFrame:
    """Evaluate frame importance using Groq + LLaMA-4 model, and flag results."""

    df = pd.read_csv(input_csv)
    llm_flags = []
    reasons = []

    for idx, row in df.iterrows():
        frame_path = row.get("path") or row.get("keyframe")
        if not frame_path or not os.path.exists(frame_path):
            llm_flags.append("not_important")
            reasons.append("Missing file")
            continue

        if row.get("feature_flag") != "important":
            llm_flags.append("not_important")
            reasons.append("Skipped due to bad visual quality")
            continue

        print(f"🧠 Evaluating LLM importance for {frame_path}...")

        try:
            base64_image = encode_image_to_base64(frame_path)

            response = groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    f"Evaluate importance of this frame: {os.path.basename(frame_path)}\n\n"
                                    "You're helping summarize educational video content.\n"
                                    "Respond ONLY in this JSON format:\n"
                                    '{ "importance": "important" or "not_important", "reason": "..." }'
                                )
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                model=MODEL_NAME
            )

            reply = response.choices[0].message.content.strip()
            match = re.search(r'{.*}', reply, re.DOTALL)
            if match:
                parsed = json.loads(match.group(0))
                llm_flags.append(parsed.get("importance", "not_important"))
                reasons.append(parsed.get("reason", "No reason provided"))
            else:
                llm_flags.append("not_important")
                reasons.append("Invalid LLM response")

        except Exception as e:
            llm_flags.append("not_important")
            reasons.append(f"Error: {str(e)}")

    df["llm_flag"] = llm_flags
    df["llm_reason"] = reasons
    df.to_csv(output_csv, index=False)
    print(f"✅ LLM flags written to {output_csv}")
    return df
