import os
import base64
from dash import Dash, dcc, html, Input, Output, State, ctx
import dash_bootstrap_components as dbc
from dotenv import load_dotenv
from concurrent.futures import ProcessPoolExecutor

from audioTranscreption.getTranscription import transcribe_audio_in_chunks
from visualExtractionEngine.keyframes import get_keyframes
from utils.cleanup_utils import clear_tmp_directory
from search.search import search_and_respond
from transformers import CLIPProcessor, CLIPModel
from langchain_google_genai import ChatGoogleGenerativeAI
from sentence_transformers import SentenceTransformer
import dash

from summeraization.summarize import load_transcript, generate_markdown_summary
from summeraization.visuals.process import run_visual_pipeline
import flask

UPLOAD_FOLDER = "tmp"

background_color = "#fcfaf6"  # soft light beige background
headers_color = "#865D03"  

upload_background_color = "#ffffff"  # light beige fill
upload_text_color = "#5a4b2c"  # dark brown text


load_dotenv()

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", trust_remote_code=True, use_safetensors=True)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

# app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])

app = Dash(__name__)

server = app.server

# @server.route("/keyframes/<path:filename>")
@server.route("/tmp/frames/keyframes/<path:filename>")
def serve_frame(filename):
    return flask.send_from_directory("tmp/frames/keyframes", filename)

app.title = "Video Upload Example"

app.layout = html.Div([
    html.Div([
        html.Div([
            html.Img(src="/assets/logo-removebg.png",  className="image-blur-appear",style={
                "height": "400px",
                "width": "400px",
                "verticalAlign": "center"
            }),
        ], style={"textAlign": "center", "marginBottom": "1px"}),

        html.P("- Upload your lecture or tutorial video", className="slide-text delay-1",style={
            "textAlign": "center",
            "fontSize": "22px",
            "fontWeight": "500",
            "color": headers_color
        }),
        html.P("- Summarize to extract the most important frames and content", className="slide-text delay-2",style={
            "textAlign": "center",
            "fontSize": "22px",
            "fontWeight": "500",
            "color": headers_color
        }),
        html.P("- Search to find answers to questions inside the video", className="slide-text delay-3",style={
            "textAlign": "center",
            "fontSize": "22px",
            "fontWeight": "500",
            "color": headers_color
        }),

        dcc.Upload(
            id="upload-video",
            children=html.Div(["Drag and Drop or ", html.A("Select a Video")]),
            style={
                "width": "50%",
                "height": "60px",
                "lineHeight": "60px",
                "border": "none",
                "borderRadius": "10px",
                "backgroundColor": upload_background_color,  # light beige fill
                "textAlign": "center",
                "margin": "20px auto",
                "color": upload_text_color,
                "fontWeight": "bold",
                "boxShadow": "0 0 8px rgba(0,0,0,0.1)",
            },
            multiple=False
        ),html.Div(id="video-status", style={"textAlign": "center", "marginTop": "20px"}),


        dcc.Input(id="query-input", type="text", placeholder="🔎 Enter your search query", style={
            "display": "none",
            "width": "100%",
            "fontSize": "18px"
        }),

        html.Button("▶️ Run Search", id="submit-query-btn", n_clicks=0, style={
            "display": "none",
            "width": "100%",
            "fontSize": "18px",
            "marginTop": "10px"
        }),

        html.Div([
            html.Button("✨ Summarize", id="summarize-btn", n_clicks=0, style={
                "margin": "5px",
                "fontSize": "18px",
                "padding": "10px 20px"
            }),
            html.Button("🔍 Search", id="search-btn", n_clicks=0, style={
                "margin": "5px",
                "fontSize": "18px",
                "padding": "10px 20px"
            }),
        ], style={"textAlign": "center", "marginTop": "20px"}),

        html.Br(),

        # dcc.Markdown(id="result-box", placeholder="📄 Output will appear here...", style={
        #     "width": "100%",
        #     "height": "200px",
        #     "fontSize": "16px",
        #     "marginTop": "20px"
        # })
        dcc.Loading(
                    id="loading",
                    type="default",  # You can use "circle", "dot", or "default"
                    children=html.Div(id="result-box", style={"whiteSpace": "pre-wrap"})
                )


    ], style={
        "maxWidth": "900px",
        "margin": "auto",
        "padding": "20px",
        "backgroundColor": background_color,
        "borderRadius": "12px",
        "boxShadow": "0 4px 12px rgba(0, 0, 0, 0.0)"
    })
], style={
    "backgroundColor": background_color,  # soft light beige background
    "minHeight": "100vh",
    "padding": "40px"
})



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



def run_search_backend(video_path, query):
    if not video_path or not query:
        return "❌ Error: Upload a video and enter a query."

    json_path, text_path = analyze_video(video_path, "groq")

    search_response, top_images = search_and_respond(
        text_path=json_path,
        image_path="tmp/frames/keyframes",
        embedding_model=SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2"),
        model=clip_model,
        processor=clip_processor,
        llm=llm,
        query=query,
        top_k=2  # or whatever number you prefer
    )

    return dcc.Markdown(search_response['answer'])

def run_summerization_backend(video_path):

    json_path, text_path = analyze_video(video_path, "groq")

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

    return dcc.Markdown(markdown, dangerously_allow_html=True)
    # return dcc.Markdown("markdown_content has been saved to summary.md", style={"whiteSpace": "pre-wrap"})  # Display the markdown content in the result box

########################################################### Upload and Callbacks ###########################################################
@app.callback(
    Output("video-status", "children"),
    Input("upload-video", "contents"),
    State("upload-video", "filename")
)
def upload_video(contents, filename):
    if contents and filename:
        try:
            content_type, content_string = contents.split(',')
            decoded = base64.b64decode(content_string)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            with open(filepath, 'wb') as f:
                f.write(decoded)
            return html.Div([
                html.P("✅ Upload Successful!", style={"color": "green", "fontWeight": "bold"}),
                html.P(f"📁 {filename}", style={"fontStyle": "italic"})
            ])
        except Exception as e:
            return html.P(f"❌ Error: {e}", style={"color": "red"})
    return ""

################################################ summarize & search into one ########################################################################

@app.callback(
    Output("result-box", "children"),
    [Input("summarize-btn", "n_clicks"),
     Input("submit-query-btn", "n_clicks")],
    [State("upload-video", "filename"),
     State("query-input", "value")]
)
def handle_output(summarize_clicks, search_clicks, filename, query):
    if not filename:
        return dash.no_update

    triggered = ctx.triggered_id

    video_path = os.path.join(UPLOAD_FOLDER, filename)

    if triggered == "summarize-btn" and summarize_clicks:
        return run_summerization_backend(video_path)
    
    elif triggered == "submit-query-btn" and search_clicks and query:
        return run_search_backend(video_path, query)
    
    return dash.no_update

################################### Show run search #############################################
@app.callback(
    Output("query-input", "style"),
    Output("submit-query-btn", "style"),
    Output("summarize-btn", "style"),
    Output("search-btn", "style"),
    Input("search-btn", "n_clicks"),
    prevent_initial_call=True
)
def toggle_search_ui(n):
    if n:
        return (
            {"display": "block", "width": "100%", "fontSize": "18px"},
            {"display": "block", "width": "100%", "fontSize": "18px", "marginTop": "10px"},
            {"display": "none"},
            {"display": "none"}
        )
    return dash.no_update

############################################# main ##########################################################

if __name__ == "__main__":
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)
