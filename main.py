from audioTranscreption.getTranscription import transcribe_audio_in_chunks
from visualExtractionEngine.keyframes import get_keyframes
from utils.cleanup_utils import clear_tmp_directory
from search.search import search_and_respond
from summeraization.summarize import load_transcript, generate_markdown_summary
from summeraization.visuals.process import run_visual_pipeline
from concurrent.futures import ProcessPoolExecutor
from transformers import CLIPProcessor, CLIPModel
from langchain_google_genai import ChatGoogleGenerativeAI
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
load_dotenv()

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", trust_remote_code=True, use_safetensors=True)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.3,
)


def analyze_video(video_path : str, transcription_provider: str = "groq"):
    
    clear_tmp_directory()    

    if transcription_provider == "groq":
        with ProcessPoolExecutor() as executor:
            t_future = executor.submit(transcribe_audio_in_chunks, video_path=video_path, provider = transcription_provider ,model= "whisper-large-v3")
            v_future = executor.submit(get_keyframes, video_path, clip_model, clip_processor)

    elif transcription_provider == "fireworks":
        with ProcessPoolExecutor() as executor:
            t_future = executor.submit(transcribe_audio_in_chunks, video_path=video_path, provider = transcription_provider ,model= "whisper-v3")
            v_future = executor.submit(get_keyframes, video_path, clip_model, clip_processor)

    else:
        return f"{transcription_provider} is not avilable / supported"

    transcription_result = t_future.result()
    return transcription_result

def main(video_path, task, transcription_provider, query=None):
    # extract frames and transcripts
    json_path, text_path = analyze_video(video_path, transcription_provider)

    # use seacrh feature
    if task == "search":
        search_response, top_images=  search_and_respond(
            text_path=json_path,
            image_path="tmp/frames/keyframes",
            embedding_model=SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2"),
            model=clip_model,
            processor=clip_processor,
            llm= llm,
            query=query,
            top_k=1,
            )
        
        return search_response, top_images
    
    elif task == "summarize":
        TRANSCRIPT_PATH = text_path
        KEYFRAMES_CSV = "tmp/frames/keyframes.csv"
        DESCRIPTIONS_CSV = "tmp/frames/descriptions.csv"
        OUTPUT_MD = "summary.md"

        print("📊 Starting visual pipeline...")
        run_visual_pipeline(KEYFRAMES_CSV, DESCRIPTIONS_CSV)

        print("📝 Generating final markdown summary...")
        transcript_text = load_transcript(TRANSCRIPT_PATH)
        markdown = generate_markdown_summary(transcript_text, DESCRIPTIONS_CSV)

        with open(OUTPUT_MD, "w", encoding="utf-8") as f:
            f.write(markdown)

        return [True, True]
    else:
        return f"{task} is not a valid task"
        

if __name__ == "__main__":
    from time import time
    task = input("summarize / search: ")
    video_path = "RawVideos\Linear Regression - Hesham Asem (720p, h264).mp4"

    start = time()
    if task == "summarize":
        _, _ = main(video_path, transcription_provider= 'groq', task=task)
    else:
        query = input("what are you searching for: ")
        search_response, top_images = main(video_path, task=task, transcription_provider= 'groq', query=query)
        print(search_response['answer'])

    end = time()

    print(f"Run in: {end-start : .2f}")
