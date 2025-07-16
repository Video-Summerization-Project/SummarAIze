# Audio Transcription

## How to Use

### 1. Clone the repository
```bash
git clone https://github.com/Video-Summerization-Project/SummarAIze
cd SummarAIze
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv
venv\Scripts\activate  # On Windows
```

### 3. Install required packages
```bash
pip install -r requirements.txt
```

### 4. Install FFmpeg
- Download FFmpeg from [https://www.gyan.dev/ffmpeg/builds/](https://www.gyan.dev/ffmpeg/builds/)
- Extract the ZIP file.
- Copy the `bin` folder of FFmpeg into this project directory (same level as `main.py`).

### 5. Create a `.env` file
In the root of the repo, create a `.env` file containing:
```
GROQ_API_KEY=your_groq_key
FIREWORKS_API_KEY=your_fireworks_key
GOOGLE_API_KEY=your_google_key
```

### 6. Configure your input video/audio
Open `main.py` and modify the `__main__` section with your file path:
```python
    task = input("summarize / search: ")
    video_path = "path/to/video.mp4"

    if task == "summarize":
        _, _ = main(video_path, transcription_provider= 'groq', task=task)
    else:
        query = input("what are you searching for: ")
        search_response, top_images = main(video_path, task=task, transcription_provider= 'groq', query=query)
        print(search_response['answer'])
```

### 7. Run the script
```bash
python main.py
```

---

## Notes
- Input video must be (`.mp4`)
- Outputs are saved in the `tmp/` directory.

---
