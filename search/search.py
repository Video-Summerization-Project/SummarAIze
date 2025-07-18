import os
from PIL import Image
from .rag import *
import shutil


# def search_images(folder_path, query, model, processor, topk=2):
#     image_files = [f for f in os.listdir(folder_path)
#                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

#     if not image_files:
#         print("❌ No image files found.")
#         return []

#     images = [Image.open(os.path.join(folder_path, f)) for f in image_files]
#     inputs = processor(text=query, images=images, return_tensors="pt", padding=True)
#     outputs = model(**inputs)
#     logits_per_image = outputs.logits_per_image
#     probs = logits_per_image

#     results = list(zip(image_files, probs.tolist()))
#     results.sort(key=lambda x: x[1], reverse=True)
#     top_filenames = [os.path.splitext(filename)[0] for filename, _ in results[:topk]]

#     return top_filenames

def chunk_text(text, chunk_size=60):
    words = text.split()
    for i in range(0, len(words), chunk_size):
        yield " ".join(words[i:i+chunk_size])

def search_images(folder_path, query, model, processor, topk=2):
    image_files = [f for f in os.listdir(folder_path)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

    if not image_files:
        print("❌ No image files found.")
        return []

    images = [Image.open(os.path.join(folder_path, f)) for f in image_files]

    max_probs = [0.0] * len(images)
    for chunk in chunk_text(query):
        inputs = processor(text=chunk, images=images, return_tensors="pt", padding=True, truncation=True)
        outputs = model(**inputs)
        logits = outputs.logits_per_image.squeeze().tolist()

        max_probs = [max(p1, p2) for p1, p2 in zip(max_probs, logits)]

    results = list(zip(image_files, max_probs))
    results.sort(key=lambda x: x[1], reverse=True)
    top_filenames = [os.path.splitext(filename)[0] for filename, _ in results[:topk]]

    return top_filenames



def generate_answer_with_context(chunks, query, llm):
    """
    Generates an answer using Gemini, based on surrounding chunks.
    """
    context = "\n\n".join(
        f"[{chunk['start_time']} - {chunk['end_time']}] {chunk['text']}" for chunk in chunks
    )

    prompt = f"""
        You are a scientific assistant helping to explain content from academic or educational videos.

        Segments:
        {context}

        Question: {query}

        Instructions:
        1. Detect the language of the segments and respond fully in that language, including the timestamp sentence .
        2. Begin with: "تم ذكر هذا الموضوع في الفيديو عند التوقيت [HH:MM:SS]" (or the equivalent in English), using correct timestamp formatting.
        3. Write the response in a clear, structured paragraph with line breaks between sentences.
        4. Take on the role of a scientific assistant: explain the topic precisely and formally, suitable for researchers or students.
        5. Focus elaboration on the core concept. Avoid excessive examples unless necessary for clarity.
        6. You may expand based on your knowledge, but do not fabricate or assume facts not stated.
        7. You will recive timestamp with seconds convert it to sutible formate [HH:MM:SS]. ex: recived 105.28 format to 00:01:45.28

        example of excpected output (can be in any language but same format):
            The video mentioned this topic at timestamp [00:09:05].
            The speaker introduces the concept of decision trees as a method for making structured decisions in machine learning tasks.
            They explain how the model splits data into branches based on feature values, eventually leading to predicted outcomes.
            This hierarchical structure makes decision trees easy to interpret, especially in classification problems.
            Beyond what's mentioned in the video, decision trees are often used as base models in ensemble methods like random forests and boosting, improving their accuracy and robustness.
        """


    response = llm.invoke(prompt)
    return {
        "answer": str(response.content).strip(),
        "chunks": chunks
    }


def clean_tmp_folder(path="tmp/search"):
    if os.path.exists(path):
        for item in os.listdir(path):
            item_path = os.path.join(path, item)
            if os.path.isfile(item_path):
                os.remove(item_path)
            else:
                shutil.rmtree(item_path)

def search_and_respond(text_path, image_path, embedding_model, model, processor, llm, query=None, top_k=1):
    clean_tmp_folder()
    print("✅ Starting")
    text = read_text(text_path)
    print("✅ Got Text")
    chunks = split_text_into_chunks(text)
    print("✅ Chunks created")
    initialize_db(embedding_model, chunks)
    print("✅ Stored in vectorDB")
    surrounding_chunks = retrieve_with_surrounding_chunks(chunks, query, embedding_model)
    print("✅ Restored surrounding_chunks")
    text_response = generate_answer_with_context(surrounding_chunks, query, llm)
    print("✅ text answer generated")

    translation_prompt = f"""
    you are an expert in translation from Arabic to English
    translate that {query} only to English
    """
    translation_response = llm.invoke(translation_prompt)
    translation = str(translation_response.content).strip()
    top_images = search_images(image_path, translation, model, processor, top_k)

    return text_response, top_images