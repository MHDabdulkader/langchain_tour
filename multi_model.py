import base64
import sys
import warnings
from dotenv import load_dotenv
import requests


# Suppress Pydantic v1 warning on Python 3.14+
warnings.filterwarnings("ignore", category=UserWarning, module="langchain_core.utils.pydantic")

# Ensure UTF-8 output on Windows terminals
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

load_dotenv()

from langchain_core.messages import HumanMessage
from langchain_openrouter import ChatOpenRouter

# ==============================================================================
# Helper: Encode local image to base64 data URL (optional, for local files)
# ==============================================================================
def encode_image_to_data_url(image_path: str, mime_type: str = "image/jpeg") -> str:
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


# ==============================================================================
# 1. Initialize Free Multimodal Model on OpenRouter
# ==============================================================================
# Why not "reflection/beam-1.0-501b-a23b"?
# -> reflection/beam is a pure text model and rejects image inputs.
#
# "openrouter/free" automatically routes to active 100% free multimodal vision
# models (such as nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free) at $0 cost!
print("[INIT] Initializing 100% Free Vision Model on OpenRouter...")
model = ChatOpenRouter(
    model="nvidia/nemotron-3.5-super-vl:free",
    max_tokens=2048,
    temperature=0.3,
)

# ==============================================================================
# 2. Build Multimodal Message
# ==============================================================================

def url_to_data_url(url: str) -> str:
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    r.raise_for_status()
    mime = r.headers.get("content-type", "image/jpeg").split(";")[0]
    return f"data:{mime};base64,{base64.b64encode(r.content).decode()}"


image_url = url_to_data_url("https://images.pexels.com/photos/40358317/pexels-photo-40358317.jpeg")

message = HumanMessage(
    content=[
        {
            "type": "text",
            "text": "Describe what you see in this image in 2-3 sentences.",
        },
        {
            "type": "image_url",
            "image_url": {
                "url": image_url,
            },
        },
    ]
)

# ==============================================================================
# 3. Invoke Model & Print Response
# ==============================================================================
print("[RUN] Sending image to free multimodal model...\n")
response = model.invoke([message])

output_text = response.content
if not output_text and hasattr(response, "additional_kwargs"):
    output_text = response.additional_kwargs.get("reasoning_content", "")

print("=" * 60)
print("RESPONSE:")
print("=" * 60)
print(output_text)