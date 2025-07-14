import time
from groq import Groq, RateLimitError
def transcribe_with_groq(client: Groq, file_path: str, model: str, language: str, timestamp_type: str):
    """
    Transcribe an audio file using the Groq Whisper API and measure elapsed time.

    Args:
        client (Groq): Initialized Groq client instance (from `groq import Groq`).
        file_path (str): Path to the audio file (must exist, supported formats: flac, wav, mp3, m4a, ogg, etc).
        model (str): Whisper model variant to use. 
        language (str): Language code for transcription (e.g., "en", "ar", etc).
        timestamp_type (str): Timestamp granularity, one of:
            - "segment": gives sentence-level timing
            - "word": gives word-level timing
        Groq supports both modes in "verbose_json" 

    Returns:
        tuple[dict, float]:
            - dict: Parsed JSON response matching Groq's "verbose_json" format.
            - float: Elapsed time in seconds for the API call.

    Raises:
        FileNotFoundError: If `file_path` doesn't exist.
        Any error raised by the Groq client (e.g., HTTP/auth issues).
    """
  
    start = time.time()
    with open(file_path, "rb") as f:
        result = client.audio.transcriptions.create(
            file=("chunk.flac", f, "audio/flac"),
            model=model,
            language=language,
            response_format="verbose_json",
            timestamp_granularities=[timestamp_type]
        )
    return result, time.time() - start


