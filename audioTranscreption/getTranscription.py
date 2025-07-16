from .audioProcessing.convert_process_audio import convert_audio_ffmpeg 
from .transcribers.transcribe_single_chunk import transcribe_single_chunk
from .audioProcessing.merge_transcripts import merge_transcripts
from .utils.save_results import save_results
from pathlib import Path
from pydub import AudioSegment
from groq import Groq
from fireworks.client.audio import AudioInference
import os


def transcribe_audio_in_chunks(video_path: Path, chunk_length: int = 600, overlap: int = 10,provider:str='fireworks',model="whisper-v3") -> dict:
    """
    Transcribe audio in chunks with overlap with Whisper via Groq API.
    
    Args:
        video_path: Path to audio file
        chunk_length: Length of each chunk in seconds
        overlap: Overlap between chunks in seconds
    
    Returns:
        dict: Containing transcription results
    
    Raises:
        ValueError: If Groq API key is not set
        RuntimeError: If audio file fails to load
    """
    
    if provider == 'groq':
          api_key = os.getenv("GROQ_API_KEY")
          client = Groq(api_key=api_key, max_retries=0)

    elif provider == 'fireworks':
          api_key = os.getenv("FIREWORKS_API_KEY")
          client = AudioInference(model=model, api_key=api_key)
    
    #print(f"\nStarting transcription of: {video_path}")


    processed_path = None
    try:
        processed_path = convert_audio_ffmpeg(video_path)
        try:
            audio = AudioSegment.from_file(processed_path, format="flac")
        except Exception as e:
            raise RuntimeError(f"Failed to load audio: {str(e)}")
        
        duration = len(audio)
        #print(f"Audio duration: {duration/1000:.2f}s")
        
        chunk_ms = chunk_length * 1000
        overlap_ms = overlap * 1000
        total_chunks = (duration // (chunk_ms - overlap_ms)) + 1
        #print(f"Processing {total_chunks} chunks...")
        
        results = []
        total_transcription_time = 0

        for i in range(total_chunks):
            start = i * (chunk_ms - overlap_ms)
            end = min(start + chunk_ms, duration)
                
            #print(f"\nProcessing chunk {i+1}/{total_chunks}")
            #print(f"Time range: {start/1000:.1f}s - {end/1000:.1f}s")
                
            chunk = audio[start:end]
            
            result, chunk_time = transcribe_single_chunk(client, chunk, i+1, total_chunks,provider=provider,model=model)
            total_transcription_time += chunk_time
            results.append((result, start))
            
        final_result = merge_transcripts(results)
        json_path = save_results(final_result, video_path)
        text_path = json_path[:-10] + ".txt"
        #print(f"\nTotal Groq API transcription time: {total_transcription_time:.2f}s")
        
        #return final_result
        return json_path, text_path
    
    finally:
        if processed_path:
            Path(processed_path).unlink(missing_ok=True)


if __name__ == "__main__":
    path = "Optimization-Day05.mp4"
    transcribe_audio_in_chunks(Path(path), chunk_length=600, overlap=10, provider='groq', model="whisper-large-v3")
        