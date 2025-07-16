from .transcribe_with_fireworks import transcribe_with_fireworks
from .transcribe_with_groq import transcribe_with_groq
import os
import tempfile, time
from requests.exceptions import HTTPError

def transcribe_single_chunk(
    client,
    chunk,
    chunk_num: int,
    total_chunks: int,
    provider: str = "groq",
    model: str = "whisper-large-v3",
    language: str = "ar",
    timestamp_type: str = "segment"
):
    """
    Transcribes a single audio chunk using the specified provider API.

    This function exports the provided AudioSegment chunk to a temporary
    FLAC file, sends it to either Groq or Fireworks Whisper transcription
    APIs, handles retries on transient errors (e.g., rate limits), and 
    returns the JSON result along with cumulative API processing time.

    Args:
        client: Initialized client object for the chosen provider.
            - For Groq: an instance of `Groq`.
            - For Fireworks: an instance of `AudioInference` or similar.
        chunk (AudioSegment): The audio segment to be transcribed.
        chunk_num (int): 1-based index of this chunk in the full sequence.
        total_chunks (int): Total number of chunks being processed.
        provider (str): Either `"groq"` or `"fireworks"` to select backend.
        model (str): Name of the Whisper model variant (e.g. `"whisper-large-v3"`).
        language (str): Language code for transcription, e.g. `"ar"`, `"en"`.
        timestamp_type (str): Timestamp granularity for output; typically `"segment"` or `"word"`.

    Returns:
        tuple:
            result (dict): The transcription JSON as returned by the provider API.
            total_time (float): Accumulated API call time (in seconds) for this chunk.

    Raises:
        ValueError: If `provider` is not one of `"groq"` or `"fireworks"`.
        RuntimeError: If transcription fails after 3 retry attempts.
        FileNotFoundError: If the exported temporary file cannot be found.
        Exception: Propagates other exceptions (e.g., network or parsing issues).

    Notes:
        - The temporary FLAC file is deleted after transcription completes or fails.
        - Retries are performed up to 3 times with delays (5s or 60s on rate limits).
        - Designed for use in pipeline loops over multiple audio chunks.
    """
    total_time = 0.0

    # Export to temporary FLAC
    with tempfile.NamedTemporaryFile(suffix=".flac", delete=False) as tf:
        chunk.export(tf.name, format="flac")
        temp_path = tf.name

    try:
        for attempt in range(3):
            try:
                if provider == "groq":
                    result, elapsed = transcribe_with_groq(client, temp_path, model, language, timestamp_type)
                elif provider == "fireworks":
                    result, elapsed = transcribe_with_fireworks(client, temp_path, model, language, timestamp_type)
                else:
                    raise ValueError(f"Unsupported provider: {provider}")

                total_time += elapsed
                #print(f"[{provider.upper()}] Chunk {chunk_num}/{total_chunks} → {elapsed:.2f}s")
                return result, total_time

            except (HTTPError, Exception) as e:
                # Handle rate limits or transient failures
                wait = 60 if "429" in str(e) else 5
                #print(f"Attempt {attempt+1} failed ({e}), retrying in {wait}s…")
                time.sleep(wait)

        raise RuntimeError(f"{provider} failed after 3 attempts")

    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass
