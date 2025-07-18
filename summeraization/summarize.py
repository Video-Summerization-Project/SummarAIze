import pandas as pd
import os
from llm.groq_model import groq_client, MODEL_NAME  # make sure this points to your Groq client/module


def load_transcript(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def generate_markdown_summary(
    transcript_text: str,
    keyframes_csv_path: str = "tmp/frames/descriptions.csv"
) -> str:

    try:
        df = pd.read_csv(keyframes_csv_path)
    except FileNotFoundError:
        print(f"❌ File not found: {keyframes_csv_path}")
        return ""

    important_df = df[
        (df.get("feature_flag") == "important") &
        (df.get("llm_flag") == "important")
    ]

    visuals = []
    for _, row in important_df.iterrows():
        image_path = row.get("path") or row.get("keyframe")
        explanation = row.get("explanation", "").strip()
        summary = row.get("summary", "").strip()
        if explanation and summary and image_path and os.path.exists(image_path):
            visuals.append({
                "path": image_path.replace("\\", "/"),
                "summary": summary,
                "explanation": explanation
            })

    visual_snippets = "\n".join([
        f"- path: {v['path']}\n  description: {v['summary'] or v['explanation']}"
        for v in visuals
    ])
    prompt = f"""
        You are an expert assistant specialized in summarizing educational videos for students.

        Your task:
        1. You will be given:
        - A transcript of a lecture (in Arabic or English).
        - A list of visual descriptions, each with a file path and what it visually contains.

        2. Your goals:
        - Organize the transcript into clear, well-structured **topics**.
        - For each topic:
            - Summarize the spoken content clearly.
            - If a visual description matches this topic, insert the image using:
            ```html
            <div style="text-align:center">
                <img src="exact path" alt="short caption" width="500"/>
            </div>
            ```
            - Use the `path` **exactly as provided** in the visuals list.
            - The `alt` text (caption) must describe the image in the **same language as the transcript**.
            - Only include images that are **relevant** to the topic being summarized.
            - not all images has to be included
            - if you want to add more information based on your knoweldge do it but be consise.

        3. Formatting Instructions:
        - Always write in the **same language as the transcript** (Arabic or English).
        - Use:
            - `## Topic Title` for each section
            - Bullet points or concise paragraphs
            - Proper spacing and formatting for readability
        - Do **not rename or invent** image names or add visuals that are not explicitly listed.
        - Use consistent image sizing by embedding visuals using HTML tags as shown above.

        Example:
        ## ما هو الانحدار الخطي؟

        الانحدار الخطي هو طريقة في تعلم الآلة تُستخدم للتنبؤ بقيمة مستمرة بناءً على بيانات مدخلة.

        - يتم استخدامه في التنبؤ بأسعار المنازل، درجات الحرارة، أو سلوك العملاء.
        - يعتمد على علاقة خطية بين المتغيرات.

        <div style="text-align:center">
        <img src="tmp/frames/keyframes/00-00-01-234.jpg" alt="رسم يوضح معادلة الانحدار" width="500"/> --> must be centered and like this
        </div>

        Transcript:
        {transcript_text}

        Visuals:
        {visual_snippets}

        Make sure to place the relevant images directly after the matching topic, not all together at the end.
        
        Respond with only the final Markdown summary document.
        """

    try:
        print("📨 Sending summary prompt to Groq LLM...")
        response = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=MODEL_NAME
        )
        print("✅ Summary received.")
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"❌ LLM summarization error: {e}")
        return "Error generating summary."
