# Agentic Object Detection Prototype

This project is a prototype of an agentic object detection workflow. A multimodal Large Language Model (LLM) is used to guide the process of analyzing an image, deciding which parts to focus on, and invoking vision tools to detect objects.

## Core Concept

The system works as follows:
1.  An input image is divided into multiple patches.
2.  For each patch, a multimodal LLM (via OpenRouter) is consulted. The LLM receives the image patch and a prompt asking whether to analyze the patch, skip it, or re-analyze with more context.
3.  Based on the LLM's decision:
    *   **ANALYZE:** A local object detection model (Hugging Face DETR) is run on the patch.
    *   **EXPAND_CONTEXT:** A larger area around the patch is extracted and then analyzed by the DETR model.
    *   **SKIP:** The patch is ignored.
4.  Detections from all analyzed patches are aggregated and displayed on the original image.

## Project Structure

```
.
├── data/                    # Sample images and output images
│   ├── input.jpg
│   └── shark.png
├── src/                     # Source code modules
│   ├── __init__.py
│   ├── config.py            # Configuration for API keys and models
│   ├── image_utils.py       # Utilities for image loading and manipulation
│   ├── main_workflow.py     # Main script-based workflow (for reference)
│   ├── openrouter_agent.py  # Handles communication with OpenRouter LLM
│   └── vision_tool_interface.py # Wrapper for the object detection model
├── agentic_object_detection_demo.ipynb  # Jupyter notebook for demonstration
├── README.md                # This file
└── requirements.txt         # Python dependencies
```

## Setup Instructions

1.  **Clone the Repository:**
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  **Create a Python Virtual Environment:**
    It's highly recommended to use a virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    Install the required Python packages using the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: Installing `torch` and `transformers` can sometimes be memory and disk intensive. Ensure you have adequate resources.*

4.  **Configure OpenRouter API Key:**
    *   Open the file `src/config.py`.
    *   You will see the following lines:
        ```python
        OPENROUTER_API_KEY = "YOUR_OPENROUTER_API_KEY_HERE"
        OPENROUTER_MULTIMODAL_MODEL = "google/gemini-2.0-flash-exp:free" # Or your preferred model
        ```
    *   Replace `"YOUR_OPENROUTER_API_KEY_HERE"` with your actual OpenRouter API key.
    *   You can also change `OPENROUTER_MULTIMODAL_MODEL` if you wish to use a different multimodal model available on OpenRouter. The default is currently set to `google/gemini-2.0-flash-exp:free`.

## Running the Demonstration with Jupyter Notebook

The primary way to test and interact with this prototype is through the `agentic_object_detection_demo.ipynb` Jupyter notebook.

1.  **Start Jupyter Notebook:**
    Ensure your virtual environment is activated. Then, from the project's root directory, run:
    ```bash
    jupyter notebook
    ```
    This will open Jupyter in your web browser.

2.  **Open the Notebook:**
    In the Jupyter interface, click on `agentic_object_detection_demo.ipynb` to open it.

3.  **Run the Cells:**
    *   The notebook is designed to be run cell by cell.
    *   Read the markdown explanations before each code cell.
    *   You can modify the **Configuration Parameters** in the notebook (e.g., `INPUT_IMAGE_PATH`, `TARGET_CLASSES`) to test with different images or settings.
    *   The first time you run cells that use the Hugging Face `transformers` library (specifically `vision_tool_interface.py`), it will download the DETR model weights (approx. 160MB). This might take some time depending on your internet connection.
    *   The notebook will display the original image, example patches, LLM decisions (printed output), and finally, the image with detected object bounding boxes.
    *   The output image with detections will also be saved in the `data` directory, prefixed with `notebook_output_`.

## Understanding the `src` Modules

*   **`config.py`**: Stores configuration variables, primarily your OpenRouter API key and the chosen LLM model.
*   **`image_utils.py`**: Contains functions for loading images, partitioning them into patches, and extracting contextual regions from images. Uses the Pillow library.
*   **`vision_tool_interface.py`**: Provides an interface to the object detection model. Currently uses `facebook/detr-resnet-50` from the Hugging Face `transformers` library to perform detections on image patches.
*   **`openrouter_agent.py`**: Manages communication with the OpenRouter API. It sends prompts (and image data if applicable) to the specified multimodal LLM and retrieves its responses.
*   **`main_workflow.py`**: A Python script that orchestrates the end-to-end agentic detection workflow. The Jupyter notebook is largely based on this script but provides a more interactive experience.

## Notes and Limitations

*   This is a prototype. The LLM's decision-making is based on simple prompting and may not always be optimal.
*   The "convolutional" or hierarchical summarization aspect (where the agent analyzes combined results from initial patches to decide on further, larger-scale analysis) is not fully implemented in this version.
*   Error handling is basic.
*   Performance can vary depending on the LLM, image size, and number of patches.
*   Ensure your OpenRouter account has credits or access to the free models if specified.
