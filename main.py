from audioTranscreption.getTranscription import transcribe_audio_in_chunks
from visualExtractionEngine.keyframes import get_keyframes
from utils.cleanup_utils import clear_tmp_directory
from concurrent.futures import ProcessPoolExecutor
from dotenv import load_dotenv
load_dotenv()

def summarize_video(video_path : str, transcription_provider: str = "groq"):
    #make sure tmp/ has no files
    clear_tmp_directory() 

    if transcription_provider == "groq":
        with ProcessPoolExecutor() as executor:
            t_future = executor.submit(transcribe_audio_in_chunks, video_path=video_path, provider = transcription_provider ,model= "whisper-large-v3")
            v_future = executor.submit(get_keyframes, video_path)

    elif transcription_provider == "fireworks":
        with ProcessPoolExecutor() as executor:
            t_future = executor.submit(transcribe_audio_in_chunks, video_path=video_path, provider = transcription_provider ,model= "whisper-v3")
            v_future = executor.submit(get_keyframes, video_path)
    
    else:
        return f"{transcription_provider} is not avilable / supported"



    clear_tmp_directory()      #uncomment this after impleminting full function to clear temp files

    return True


if __name__ == "__main__":
    from time import time
    
    start = time()

    video_path = "RawVideos\Linear Regression - Hesham Asem (720p, h264).mp4"
    summarize_video(video_path, transcription_provider= 'groq')

    end = time()
    print(f"Run in: {end-start : .2f}")
