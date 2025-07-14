from audioTranscreption.getTranscription import transcribe_audio_in_chunks
from visualExtractionEngine.keyframes import get_keyframes
from utils.cleanup_utils import clear_tmp_directory
from concurrent.futures import ProcessPoolExecutor
from dotenv import load_dotenv
load_dotenv()

def summarize_video(video_path : str, transcription_provider: str = "groq"):

    with ProcessPoolExecutor() as executor:
        t_future = executor.submit(transcribe_audio_in_chunks, video_path=video_path, provider = transcription_provider ,model= "whisper-large-v3")
        v_future = executor.submit(get_keyframes, video_path)


        #clear_tmp_directory()      #uncomment this after impleminting full function to clear temp files

    return True


if __name__ == "__main__":
    from time import time
    
    start = time()

    video_path = "RawVideos\Linear Regression - Hesham Asem (720p, h264).mp4"
    summarize_video(video_path)

    end = time()
    print(f"Run in: {end-start : .2f}")
