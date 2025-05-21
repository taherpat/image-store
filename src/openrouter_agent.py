import base64
import requests
import json
from PIL import Image
import io
from . import config # To access OPENROUTER_API_KEY and model

def _image_to_base64(image: Image.Image) -> str | None:
    """
    Converts a Pillow Image object to a base64 encoded string (JPEG format).

    Args:
        image (PIL.Image.Image): The Pillow Image object to convert.

    Returns:
        str | None: The base64 encoded string, or None if an error occurs.
    """
    try:
        if image.mode == 'RGBA': # Convert RGBA to RGB to avoid issues with JPEG saving
            image = image.convert('RGB')
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return img_str
    except Exception as e:
        print(f"Error converting image to base64: {e}")
        return None

def get_agent_response(prompt: str, image: Image.Image = None) -> str:
    """
    Gets a response from the OpenRouter multimodal agent.

    Args:
        prompt (str): The text prompt to send to the agent.
        image (PIL.Image.Image, optional): An optional Pillow Image object to send.

    Returns:
        str: The agent's text response, or an error message if an error occurs.
    """
    if not config.OPENROUTER_API_KEY or config.OPENROUTER_API_KEY == "YOUR_OPENROUTER_API_KEY_HERE":
        return "Error: OpenRouter API key not configured in src/config.py. Please set it to your actual key."

    headers = {
        "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    messages_content = []

    if image:
        base64_image_string = _image_to_base64(image)
        if base64_image_string:
            messages_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image_string}"
                }
            })
        else:
            print("Warning: Image provided but conversion to base64 failed. Proceeding without image.")
            # Optionally, return an error here instead of proceeding without image
            # return "Error: Image conversion to base64 failed."
    
    messages_content.append({"type": "text", "text": prompt})

    data = {
        "model": config.OPENROUTER_MULTIMODAL_MODEL,
        "messages": [
            {
                "role": "user",
                "content": messages_content
            }
        ]
    }

    # If only text is present and the model expects a simple string for content
    if not image and config.OPENROUTER_MULTIMODAL_MODEL in ["anthropic/claude-3-haiku", "google/gemini-flash-1.5"]: # Example check
         # For some models, if no image, content might just be a string not a list
         # However, the list format with a single text item usually works for most advanced models
         pass # Keep as list of content blocks for consistency, it's generally supported.

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            data=json.dumps(data),
            timeout=60  # Adding a timeout
        )
        response.raise_for_status()  # Raises an HTTPError for bad responses (4XX or 5XX)
        
        response_json = response.json()
        
        if response_json.get("choices") and len(response_json["choices"]) > 0:
            message = response_json["choices"][0].get("message", {})
            content = message.get("content")
            if content:
                return content
            else:
                return f"Error: No content in agent's response. Full response: {response_json}"
        else:
            return f"Error: Unexpected response format from OpenRouter. Full response: {response_json}"

    except requests.exceptions.HTTPError as http_err:
        return f"HTTP error occurred: {http_err}. Response: {response.text}"
    except requests.exceptions.RequestException as req_err:
        return f"Request error occurred: {req_err}"
    except json.JSONDecodeError:
        return f"Error decoding JSON response from OpenRouter. Response: {response.text if 'response' in locals() else 'No response object'}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"
