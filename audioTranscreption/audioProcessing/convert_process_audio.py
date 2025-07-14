import subprocess
import os

def convert_audio_ffmpeg(input_path: str) -> str:
    """
    Converts the given audio or video file to a mono FLAC file at 16kHz,
    with filters applied for speech enhancement.

    Args:
        input_path (str): Path to the input audio or video file.

    Returns:
        str: Path to the converted FLAC file.
    """

    if not input_path:
        raise FileNotFoundError(f"Input file not found: {input_path}")

    base_name = os.path.splitext(os.path.basename(input_path))[0] + '.flac'
    output_dir = "tmp"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, base_name)

    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-ac", "1",                        # mono channel
        "-ar", "16000",                   # 16kHz sample rate
        "-sample_fmt", "s16",             # 16-bit signed PCM
        "-vn",                             # remove video
        "-af", "highpass=f=200, lowpass=f=3000, dynaudnorm",  # speech cleanup
        "-map", "0:a",                     # select only audio stream
        "-c:a", "flac",                    # encode as FLAC
        output_path
    ]

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if result.returncode != 0:
        #print("[✘] ffmpeg conversion failed:")
        #print(result.stderr)
        raise RuntimeError("Audio conversion failed.")

    #print(f"[✔] Audio converted and cleaned → {output_path}")
    return output_path


