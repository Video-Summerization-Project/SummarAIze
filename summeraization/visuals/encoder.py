import base64
from PIL import Image
from io import BytesIO


def encode_image_to_base64(image_path: str) -> str:
    """
    Load an image and encode it to a base64 JPEG string for use with Groq API.
    """
    with Image.open(image_path) as img:
        buffer = BytesIO()
        img.save(buffer, format="JPEG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
