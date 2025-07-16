import re
import os
from typing import Dict
from llm.groq_model import groq_client, MODEL_NAME
from summeraization.visuals.encoder import encode_image_to_base64


def describe_frame(image_path: str) -> Dict[str, str]:
    """
    Use Groq's LLaMA 4 model to describe the visual content of a frame image.

    Args:
        image_path: Local path to the image file.

    Returns:
        {
            "explanation": "...",
            "summary": "..."
        }
    """
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return {
            "explanation": "Missing file.",
            "summary": "Missing file."
        }

    try:
        base64_image = encode_image_to_base64(image_path)
        image_name = os.path.basename(image_path)

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"You are an expert in visual understanding of educational content.\n\n"
                            f"Analyze the image `{image_name}` and respond strictly in this format:\n\n"
                            "Explanation: ...\nSummary: ...\n\n"
                            "Guidelines:\n"
                            "- Do not include decorative elements.\n"
                            "- Avoid raw OCR or copying text.\n"
                            "- Be concise.\n"
                            "- If the image is in Arabic, respond in Arabic; otherwise, use English."
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
        ]

        response = groq_client.chat.completions.create(
            messages=messages,
            model=MODEL_NAME
        )

        reply = response.choices[0].message.content.strip()

        def extract_field(label):
            match = re.search(rf'{label}:\s*(.*?)\s*(?=\n\w+:|$)', reply, re.DOTALL)
            return match.group(1).strip() if match else "N/A"

        return {
            "explanation": extract_field("Explanation"),
            "summary": extract_field("Summary")
        }

    except Exception as e:
        print(f"❌ Error describing {image_name}: {e}")
        return {
            "explanation": "Failed to generate explanation.",
            "summary": "Failed to generate summary."
        }
