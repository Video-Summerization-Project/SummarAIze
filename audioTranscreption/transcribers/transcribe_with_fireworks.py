import time, os, requests
from pathlib import Path
from fireworks.client.audio import AudioInference

def transcribe_with_fireworks(
    client: AudioInference,
    file_path: str,
    model: str,
    language: str,
    timestamp_type: str = 'segment',
    vad_model: str = 'silero',
    temperature: float = 0.0,
    endpoint: str = "https://audio-prod.us-virginia-1.direct.fireworks.ai/v1/audio/transcriptions"
) -> tuple[dict, float]:
    """
    Transcribes audio using Fireworks.ai Whisper API with support for segment/word-level timestamps.

    Args:
        client: initialized AudioInference instance (provides .api_key)
        file_path: Path to the FLAC file (must exist)
        model: Whisper model name
        language: language code ('en', 'ar', etc.)
        timestamp_type: list e.g. ['segment'], ['word'], or both
        vad_model: VAD model to use (default 'silero')
        temperature: float, temperature sampling
        endpoint: transcription API endpoint

    Returns:
        tuple of (transcription JSON dict, elapsed time in seconds)

    Raises:
        FileNotFoundError: If `file_path` doesn't exist.
        Any error raised by the fireworks client (e.g., HTTP/auth issues)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    start = time.time()
    with open(file_path, "rb") as f:
        resp = requests.post(
            endpoint,
            headers={"Authorization": f"Bearer {client.api_key}"},
            files={"file": f},
            data={
                "model": model,
                "language": language,
                "vad_model": vad_model,
                "temperature": str(temperature),
                "timestamp_granularities": [timestamp_type],
                "response_format": "verbose_json"

            }
        )

    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        print(f"[✘] Error {resp.status_code}: {resp.text}")
        raise RuntimeError("Transcription failed.") from e

    elapsed = time.time() - start
    print(f"[✔] Transcription succeeded in {elapsed:.2f}s")

    return resp.json(), elapsed
