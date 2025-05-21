from . import image_utils
from . import vision_tool_interface
from . import openrouter_agent
from . import config # For API key check and potentially other configs
from PIL import ImageDraw, ImageFont # For drawing results later
import os

def main():
    # 1. Initial Setup
    if not config.OPENROUTER_API_KEY or config.OPENROUTER_API_KEY == "YOUR_OPENROUTER_API_KEY_HERE":
        print("Error: OpenRouter API key not configured in src/config.py. Please set it to your actual key and run again.")
        return

    input_image_path = "data/input.jpg" # Example image
    if not os.path.exists(input_image_path):
        print(f"Error: Input image not found at {input_image_path}. Please ensure it exists.")
        # Attempt to list files in data directory to help diagnose
        if os.path.exists("data"):
            print(f"Files in data/ directory: {os.listdir('data')}")
        else:
            print("Error: data/ directory does not exist.")
        return

    target_classes = ["person", "car", "dog", "cat", "bicycle", "traffic light", "stop sign"] # Expanded example
    num_rows = 3
    num_cols = 3
    expansion_factor = 1.5

    # 2. Load and Partition Image
    print(f"Loading image: {input_image_path}")
    original_image = image_utils.load_image(input_image_path)
    if original_image is None:
        print(f"Failed to load image: {input_image_path}")
        return
    
    print(f"Partitioning image into {num_rows}x{num_cols} patches...")
    patches_info = image_utils.partition_image(original_image, num_rows, num_cols)
    if not patches_info:
        print("Failed to partition image.")
        return
    print(f"Image partitioned into {len(patches_info)} patches.")

    # 3. Main Loop through Patches
    all_detections = [] # To store detections from all patches, relative to original image

    for i, patch_info in enumerate(patches_info):
        patch_image = patch_info['patch_image']
        patch_coords = patch_info['coords'] # (left, upper, right, lower) relative to original
        
        print(f"\nProcessing Patch {i+1}/{len(patches_info)} - Coords: {patch_coords}")

        # Construct initial prompt for the LLM agent
        prompt = (
            f"You are an object detection assistant. This image patch is from coordinates {patch_coords} of a larger image. "
            f"Your task is to identify if any of the following target objects might be present in THIS SPECIFIC PATCH: {', '.join(target_classes)}. "
            f"Based on the visual information in this patch, do you think it's worth running a detailed object detection model on it? "
            f"Respond with only one of these keywords: 'ANALYZE' if yes, 'SKIP' if no. "
            f"If the patch is ambiguous (e.g., shows only a small part of a potential object, like a wheel of a car) such that the full object might be outside this patch but nearby, respond with 'EXPAND_CONTEXT'. "
            f"Your response must be ONLY one of these three keywords."
        )

        # Call the OpenRouter agent
        print(f"  Sending patch to OpenRouter agent (model: {config.OPENROUTER_MULTIMODAL_MODEL})...")
        agent_decision_text = openrouter_agent.get_agent_response(prompt, image=patch_image)
        
        # Sanitize and simplify the agent's response to one of the keywords
        if isinstance(agent_decision_text, str):
            cleaned_response = agent_decision_text.strip().upper()
            if "ANALYZE" in cleaned_response:
                decision = "ANALYZE"
            elif "EXPAND_CONTEXT" in cleaned_response:
                decision = "EXPAND_CONTEXT"
            elif "SKIP" in cleaned_response:
                decision = "SKIP"
            else:
                decision = "SKIP" # Default to SKIP if no clear keyword found
                print(f"  Patch {patch_coords}: Agent raw decision: '{agent_decision_text}'. Could not parse a clear keyword, defaulting to SKIP.")
        else: # If the response isn't a string (e.g. error message from API)
            decision = "SKIP"
            print(f"  Patch {patch_coords}: Agent response was not a string: '{agent_decision_text}'. Defaulting to SKIP.")

        print(f"  Patch {patch_coords}: Agent decision: {decision}")

        if decision == "ANALYZE":
            print(f"  Action: ANALYZE - Running object detection on current patch {patch_coords}...")
            detections = vision_tool_interface.detect_objects(patch_image, target_classes)
            if detections:
                print(f"    Found {len(detections)} objects in patch {patch_coords}.")
                for det in detections:
                    # Convert patch-relative box [x_patch, y_patch, w, h] to original image coordinates
                    det['box'][0] += patch_coords[0]  # x_orig = x_patch + patch_left
                    det['box'][1] += patch_coords[1]  # y_orig = y_patch + patch_top
                    all_detections.append(det)
            else:
                print(f"    No objects found in patch {patch_coords} by vision tool.")

        elif decision == "EXPAND_CONTEXT":
            print(f"  Action: EXPAND_CONTEXT - Getting contextual patch and running object detection...")
            contextual_patch_img, contextual_coords = image_utils.get_contextual_patch(original_image, patch_coords, expansion_factor)
            print(f"    Contextual patch coords (original image): {contextual_coords}")
            
            detections = vision_tool_interface.detect_objects(contextual_patch_img, target_classes)
            if detections:
                print(f"    Found {len(detections)} objects in contextual patch {contextual_coords}.")
                for det in detections:
                    # Convert contextual-patch-relative box [x_context, y_context, w, h] to original image coordinates
                    det['box'][0] += contextual_coords[0]  # x_orig = x_context + contextual_patch_left
                    det['box'][1] += contextual_coords[1]  # y_orig = y_context + contextual_patch_top
                    all_detections.append(det)
            else:
                print(f"    No objects found in contextual patch {contextual_coords} by vision tool.")
                
        elif decision == "SKIP":
            print(f"  Action: SKIP - Skipping detailed analysis for patch {patch_coords} based on agent decision.")
            pass
    
    print("\nFinished processing all patches.")

    # 4. Summarization Step (Optional - keeping as placeholder for now)
    # if all_detections:
    #     # This is where you could send `all_detections` (with global coordinates) to an LLM
    #     # to get a summary of the entire image's content.
    #     # For example:
    #     # summary_prompt_parts = [f"Object: {d['label']}, Location: (x={d['box'][0]}, y={d['box'][1]}, w={d['box'][2]}, h={d['box'][3]})" for d in all_detections]
    #     # summary_prompt = f"The following objects were detected in the image: {'; '.join(summary_prompt_parts)}. Provide a brief human-readable summary of the scene."
    #     # final_summary = openrouter_agent.get_agent_response(summary_prompt) # No image needed for this call
    #     # print(f"\nOverall Image Summary:\n{final_summary}")
    #     pass
    # else:
    #     # print("\nNo objects detected in the image after processing all patches for summarization.")
    #     pass

    # 5. Display/Save Results
    if all_detections:
        print(f"\nTotal objects detected in the image: {len(all_detections)}")
        draw_image = original_image.copy()
        draw = ImageDraw.Draw(draw_image)
        try:
            font = ImageFont.load_default()
        except IOError:
            print("Default font not found. Using a fallback.")
            font = None # Or specify a path to a known font if available

        for det in all_detections:
            box = det['box']  # These are now original image coordinates [x, y, w, h]
            label = f"{det['label']}: {det['score']:.2f}"
            
            # Define rectangle for drawing [left, top, right, bottom]
            rect = [box[0], box[1], box[0] + box[2], box[1] + box[3]]
            draw.rectangle(rect, outline="red", width=3)
            
            # Adjust text position if it goes off-image (simple adjustment)
            text_x = box[0]
            text_y = box[1] - 15 if box[1] - 15 > 0 else box[1] + 5 # move above, or below if too close to top
            
            draw.text((text_x, text_y), label, fill="red", font=font)

        base_name = os.path.basename(input_image_path)
        output_image_path = os.path.join("data", "output_" + base_name)
        
        try:
            draw_image.save(output_image_path)
            print(f"Processed image with detections saved to: {output_image_path}")
            # draw_image.show() # Optional: display the image
        except Exception as e:
            print(f"Error saving or showing image: {e}")
            
    else:
        print("\nNo objects were detected in the image after processing all patches.")

if __name__ == "__main__":
    main()
