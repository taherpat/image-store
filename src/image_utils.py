from PIL import Image

def load_image(image_path):
    """
    Loads an image from the given path, converts it to RGB, and returns the Pillow Image object.

    Args:
        image_path (str): The path to the image file.

    Returns:
        PIL.Image.Image: The loaded image object in RGB format, or None if an error occurs.
    """
    try:
        img = Image.open(image_path)
        img = img.convert('RGB')
        return img
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def partition_image(image, num_rows, num_cols):
    """
    Partitions the given image into a grid of patches.

    Args:
        image (PIL.Image.Image): The Pillow Image object to partition.
        num_rows (int): The number of rows in the grid.
        num_cols (int): The number of columns in the grid.

    Returns:
        list: A list of dictionaries, where each dictionary contains:
              'patch_image' (PIL.Image.Image): The patch image object.
              'coords' (tuple): A tuple (left, upper, right, lower) representing the
                                coordinates of the patch in the original image.
    """
    img_width, img_height = image.size
    patch_width = img_width // num_cols
    patch_height = img_height // num_rows
    patches = []

    for i in range(num_rows):
        for j in range(num_cols):
            left = j * patch_width
            upper = i * patch_height
            # For the last column, extend to the image width
            right = (j + 1) * patch_width if j < num_cols - 1 else img_width
            # For the last row, extend to the image height
            lower = (i + 1) * patch_height if i < num_rows - 1 else img_height
            
            patch_image = image.crop((left, upper, right, lower))
            patches.append({'patch_image': patch_image, 'coords': (left, upper, right, lower)})
            
    return patches

def get_image_size(image):
    """
    Gets the dimensions of the given Pillow Image object.

    Args:
        image (PIL.Image.Image): The Pillow Image object.

    Returns:
        tuple: A tuple (width, height) representing the image dimensions.
    """
    return image.size

def get_contextual_patch(original_image, patch_coords, expansion_factor):
    """
    Extracts a contextual patch from the original image, expanding from a given patch's coordinates.

    Args:
        original_image (PIL.Image.Image): The Pillow Image object of the full image.
        patch_coords (tuple): A tuple (left, upper, right, lower) defining the box of the original patch.
        expansion_factor (float): Factor to expand the patch (e.g., 1.5 for 50% expansion).

    Returns:
        tuple: A tuple containing:
               - PIL.Image.Image: The cropped contextual patch.
               - tuple: The new coordinates (left, upper, right, lower) of the contextual patch.
    """
    original_left, original_upper, original_right, original_lower = patch_coords
    patch_width = original_right - original_left
    patch_height = original_lower - original_upper

    center_x = original_left + patch_width / 2
    center_y = original_upper + patch_height / 2

    new_width = patch_width * expansion_factor
    new_height = patch_height * expansion_factor

    new_left = center_x - new_width / 2
    new_upper = center_y - new_height / 2
    new_right = center_x + new_width / 2
    new_lower = center_y + new_height / 2

    # Clip coordinates to image boundaries
    img_width, img_height = original_image.size
    new_left_clipped = max(0, int(new_left))
    new_upper_clipped = max(0, int(new_upper))
    new_right_clipped = min(img_width, int(new_right))
    new_lower_clipped = min(img_height, int(new_lower))

    # Ensure clipped coordinates do not result in a zero-area patch if possible
    if new_left_clipped >= new_right_clipped:
        if new_right_clipped == img_width: # if anchored at the right edge
             new_left_clipped = max(0, new_right_clipped -1) # make it at least 1 pixel wide if not at 0
        else: # if not anchored at the right edge or new_left was > new_right
             new_right_clipped = new_left_clipped + 1 if new_left_clipped < img_width else new_left_clipped
             if new_right_clipped > img_width : # if it goes over, clip it.
                  new_right_clipped = img_width
                  new_left_clipped = img_width -1 if img_width > 0 else 0


    if new_upper_clipped >= new_lower_clipped:
        if new_lower_clipped == img_height: # if anchored at the bottom edge
            new_upper_clipped = max(0, new_lower_clipped -1) # make it at least 1 pixel high if not at 0
        else: # if not anchored at the bottom edge or new_upper was > new_lower
            new_lower_clipped = new_upper_clipped + 1 if new_upper_clipped < img_height else new_upper_clipped
            if new_lower_clipped > img_height: # if it goes over, clip it.
                new_lower_clipped = img_height
                new_upper_clipped = img_height -1 if img_height > 0 else 0


    contextual_patch_img = original_image.crop((new_left_clipped, new_upper_clipped, new_right_clipped, new_lower_clipped))
    final_coords = (new_left_clipped, new_upper_clipped, new_right_clipped, new_lower_clipped)

    return contextual_patch_img, final_coords
