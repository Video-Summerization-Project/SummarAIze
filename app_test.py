import dash
from dash import dcc, html, Input, Output, State
import base64
import os

app = dash.Dash(__name__)
app.title = "Video Upload Example"

UPLOAD_FOLDER = "tmp"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.layout = html.Div([
    html.H2("Upload your video"),
    dcc.Upload(
        id="upload-video",
        children=html.Div(["Drag and Drop or ", html.A("Select a Video")]),
        style={
            "width": "50%",
            "height": "60px",
            "lineHeight": "60px",
            "borderWidth": "1px",
            "borderStyle": "dashed",
            "borderRadius": "5px",
            "textAlign": "center",
            "margin": "20px auto"
        },
        multiple=False
    ),
    html.Div(id="video-status", style={"textAlign": "center", "marginTop": "20px"})
])

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

if __name__ == "__main__":
    app.run(debug=True)
