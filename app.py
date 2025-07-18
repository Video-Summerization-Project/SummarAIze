from audioTranscreption.getTranscription import transcribe_audio_in_chunks
from visualExtractionEngine.keyframes import get_keyframes
from utils.cleanup_utils import clear_tmp_directory
from search.search import search_and_respond
from concurrent.futures import ProcessPoolExecutor
from transformers import CLIPProcessor, CLIPModel
from langchain_google_genai import ChatGoogleGenerativeAI
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import gradio as gr

load_dotenv()

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", trust_remote_code=True, use_safetensors=True)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)


def analyze_video(video_path: str, transcription_provider: str = "groq"):
    clear_tmp_directory()
    with ProcessPoolExecutor() as executor:
        if transcription_provider == "groq":
            t_future = executor.submit(transcribe_audio_in_chunks, video_path=video_path, provider="groq", model="whisper-large-v3")
        elif transcription_provider == "fireworks":
            t_future = executor.submit(transcribe_audio_in_chunks, video_path=video_path, provider="fireworks", model="whisper-v3")
        else:
            return f"{transcription_provider} not supported"
        v_future = executor.submit(get_keyframes, video_path, clip_model, clip_processor)
    return t_future.result()


def run_search(video_file, query):
    if not video_file or not query:
        raise gr.Error("Upload a video and enter a query.")
    
    text_path = analyze_video(video_file, "groq")
    response, images = search_and_respond(
        text_path=text_path,
        image_path="tmp/frames/keyframes",
        embedding_model=SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2"),
        model=clip_model,
        processor=clip_processor,
        llm=llm,
        query=query,
        top_k=1
    )
    return response.get("answer", "❌ No answer found.")


def process_uploaded_video_summarize(video_file):
    if not video_file:
        raise gr.Error("Please upload a video first.")
    return analyze_video(video_file)


# Show/hide logic
def show_query_box():
    return (
        gr.update(visible=True),   # query_input
        gr.update(visible=True),   # submit_query_btn
        gr.update(visible=False),  # summarize_btn
        gr.update(visible=False),  # search_btn
    )


with gr.Blocks(theme=gr.themes.Default(primary_hue=gr.themes.colors.zinc)) as demo:

    gr.HTML(
        """
        <style>
            body {
                font-family: 'Segoe UI', sans-serif;
                background-color: #ffffff;
            }

            #app-header {
                text-align: center;
                margin-bottom: 30px;
            }

            #app-header h1 {
                font-size: 2.8rem;
                color: #b07600;
                font-weight: bold;
                margin-bottom: 10px;
            }

            #app-header img {
                width: 60px;
                margin-top: 10px;
            }

            #app-description {
                color: #292a2b;
                font-size: 1.05rem;
                line-height: 1.5;
                margin-top: 15px;
            }

            Button.svelte-1ipelgc {
                background-color: #ffffff !important;
                color: #b07600 !important;
                border: 2px solid #b07600 !important;
                border-radius: 8px !important;
                font-size: 1rem;
                font-weight: 600;
                padding: 10px 20px !important;
                transition: background-color 0.2s ease, color 0.2s ease;
            }

            Button.svelte-1ipelgc:hover {
                background-color: #b07600 !important;
                color: #ffffff !important;
            }

            .gr-box, .gr-file {
                border-radius: 10px !important;
                background-color: #fff !important;
                border: 1px solid #ddd !important;
                padding: 15px;
            }

            label {
                font-weight: 600 !important;
                color: #091729 !important;
            }

            textarea, input[type="text"] {
                background-color: #f0f0f0 !important;
                border-radius: 10px !important;
                padding: 10px;
                color: #000000 !important;
            }
        </style>
        """
    )

    with gr.Column(elem_id="centered-content"):
        with gr.Row():
            gr.Image("assets/logo-removebg.png",width=300, height=300, show_label=False, show_download_button=False)  


    # with gr.Blocks(theme=gr.themes.Default(primary_hue=gr.themes.colors.red)) as demo:
    #     gr.Button("Click Me")
        
    gr.HTML(
        """
        <div id="app-header">
            <div id="app-description">
            <h1>sumurAIze</h1>
                <p>Upload your lecture or tutorial video</p>
                <p><b>Summarize</b> to extract the most important frames and content</p>
                <p><b>Search</b> to find answers to questions inside the video</p>
            </div>
        </div>
        """
    )

    with gr.Column(elem_id="centered-content"):
        video_upload = gr.File(label="🎥 Upload Video", file_types=["video"], type="filepath")
        query_input = gr.Textbox(label="🔎 Enter your search query", visible=False)
        submit_query_btn = gr.Button("▶️ Run Search", variant="secondary", visible=False)
        with gr.Row():
            summarize_btn = gr.Button("✨ Summarize", variant="primary", scale=0.5)
            search_btn = gr.Button("🔍 Search", variant="primary", scale=0.5)


        result_box = gr.Textbox(label="📄 Output", lines=8, interactive=False)

    summarize_btn.click(fn=process_uploaded_video_summarize, inputs=[video_upload], outputs=[result_box])
    search_btn.click(
        fn=show_query_box,
        inputs=[],
        outputs=[query_input, submit_query_btn, summarize_btn, search_btn]
    )
    submit_query_btn.click(fn=run_search, inputs=[video_upload, query_input], outputs=[result_box])

if __name__ == "__main__":
    demo.launch()
