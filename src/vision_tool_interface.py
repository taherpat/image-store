from PIL import Image
from transformers import DetrImageProcessor, DetrForObjectDetection
import torch

# Initialize model and processor
try:
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
except Exception as e:
    print(f"Error loading Hugging Face model or processor: {e}")
    # Set to None so that functions using them can check and fail gracefully
    processor = None
    model = None

def detect_objects(image_patch: Image.Image, target_classes: list[str] = None):
    """
    Detects objects in the given image patch using a pre-trained DETR model.

    Args:
        image_patch (PIL.Image.Image): The Pillow Image object (RGB format).
        target_classes (list[str], optional): A list of class names to filter for.
                                             If None or empty, detects all classes.

    Returns:
        list: A list of dictionaries, where each dictionary represents a detected object
              and contains 'box' ([x, y, w, h]), 'label' (class_name), and 'score'.
              Returns an empty list if no objects are detected, if the model failed to load,
              or an error occurs.
    """
    if not processor or not model:
        print("Error: Hugging Face model or processor not loaded. Cannot perform detection.")
        return []

    if not image_patch:
        print("Error: Input image_patch is None.")
        return []

    try:
        # Ensure image is RGB
        if image_patch.mode != 'RGB':
            image_patch = image_patch.convert('RGB')

        inputs = processor(images=image_patch, return_tensors="pt")
        outputs = model(**inputs)

        # Post-process results
        # target_sizes expects [height, width]
        results = processor.post_process_object_detection(outputs, threshold=0.7, target_sizes=torch.tensor([image_patch.size[::-1]]))

        detections = []
        if results and len(results) > 0:
            for score, label, box in zip(results[0]["scores"], results[0]["labels"], results[0]["boxes"]):
                class_name = model.config.id2label[label.item()]

                # Filter by target_classes if provided
                if target_classes and class_name not in target_classes:
                    continue

                xmin, ymin, xmax, ymax = box.tolist()
                w = xmax - xmin
                h = ymax - ymin
                
                detection = {
                    'box': [int(xmin), int(ymin), int(w), int(h)],
                    'label': class_name,
                    'score': score.item()
                }
                detections.append(detection)
        
        return detections

    except Exception as e:
        print(f"Error during object detection: {e}")
        return []
