
# pip install ultralytics==8.0.196


import ultralytics
ultralytics.checks()

from ultralytics import YOLO


from IPython.display import display, Image
import os
import cv2
import supervision as sv
import torch
import supervision as sv
import numpy as np
from PIL import Image
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import json

# pip install -q 'git+https://github.com/facebookresearch/segment-anything.git'
# pip install supervision
# mkdir -p {HOME}/weights
# wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -P {HOME}/weights

HOME = os.getcwd()
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = "vit_h"
CHECKPOINT_PATH = os.path.join(HOME, "weights", "sam_vit_h_4b8939.pth")
# print("HOME:", HOME)

# Accessing the HOME environment variable
model_path = os.path.join(HOME, "content", "new.pt")
print(model_path)

# Load the YOLO model|
model = YOLO(model_path)

model.names
class_names = model.names
class_names

# Commented out IPython magic to ensure Python compatibility.
# %pip install supervision
# use different location here

test_path = os.path.join(HOME, "downloaded_images")
image_boxes=[]
images=[]
image_labels=[]

# List all files in the test_path directory and print each one

for image_file in os.listdir(test_path):
    # print("im",image_file)
    image_path = f"{test_path}/{image_file}"
    # Load the image using OpenCV
    image = cv2.imread(image_path)
    # print("hello",image_path)

  # Get image dimensions (height, width, channels)
    height, width, channels = image.shape
    results = model.predict(source=f"{test_path}/{image_file}", conf=0.25, imgsz=(height, width))
    images.append(image_file)
    image_boxes.append((results[0].boxes.xyxy).cpu().tolist())
    labels=[]

    # print(image_file)
    for i in range(len(results[0].boxes.cls)):
      class_idx = int(results[0].boxes.cls[i])  # Get class index as integer
      class_name = class_names[class_idx]       # Look up the class name
      confidence = results[0].boxes.conf[i].item()  # Confidence score
      labels.append(class_name)

      # print(f"Bounding Box {i + 1}: Class = {class_name}, Confidence = {confidence} \n bounding box={results[0].boxes.xyxy}")

    image_labels.append(labels)


print(CHECKPOINT_PATH, "; exist:", os.path.isfile(CHECKPOINT_PATH))

sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)

mask_predictor = SamPredictor(sam)

segment_path= os.path.join(HOME, "segmented_images")

mask_strength = 4

for k in range(len(images)):
    image_path = f"{test_path}/{images[k]}"
    image_bgr = cv2.imread(image_path)
    boxes = image_boxes[k]
    label = image_labels[k]

    for i in range(len(boxes)):
        box = np.array(boxes[i])
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mask_predictor.set_image(image_rgb)

        masks, scores, logits = mask_predictor.predict(
            box=box,
            multimask_output=True
        )

        # Define color based on label
        color_select = sv.Color(r=255, g=0, b=0) if label[i] == "weed" else sv.Color(r=0, g=255, b=0)

        # Create annotators
        box_annotator = sv.BoxAnnotator(color=color_select)
        mask_annotator = sv.MaskAnnotator(color=color_select, color_lookup=sv.ColorLookup.INDEX)

        # Create dummy class_id for detections
        class_ids = np.zeros(len(masks), dtype=int)

        detections = sv.Detections(
            xyxy=sv.mask_to_xyxy(masks=masks),
            mask=masks,
            class_id=class_ids
        )
        detections = detections[detections.area == np.max(detections.area)]

        # Annotate the images
        source_img = box_annotator.annotate(scene=image_bgr.copy(), detections=detections)
        for _ in range(mask_strength):
            image_bgr = mask_annotator.annotate(scene=image_bgr.copy(), detections=detections)

    # Save the final annotated image
    output_path = f"{segment_path}/{images[k]}"
    cv2.imwrite(output_path, image_bgr)

    print(f"Saved segmented image to {output_path}")

def convert_jpeg_to_png(directory):
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.jpeg', '.jpg')):
            file_path = os.path.join(directory, filename)
            png_path = os.path.splitext(file_path)[0] + '.png'  # Use same path with .png extension

            # Open the JPEG image and convert to PNG
            with Image.open(file_path) as img:
                img.save(png_path)

            # Remove the original JPEG file
            os.remove(file_path)
            print(f"Converted and replaced {file_path} with {png_path}")

# Specify the path to your directory
directory = segment_path

# Call the function to convert and replace all JPEG images in the directory with PNG
convert_jpeg_to_png(directory)

# Iterate through all files in the current directory
for filename in os.listdir(segment_path):
    if filename.endswith('.png'):  # Process only PNG files
        # Open the image
        img_path = os.path.join(segment_path, filename)
        img = Image.open(img_path)

        # Convert RGBA to RGB if the image has an alpha channel
        if img.mode == 'RGBA':
            rgb_img = img.convert('RGB')

            # Save the converted image, overwriting the original
            rgb_img.save(segment_path+filename)

            # print(f"Converted {filename} from RGBA to RGB")

        # Close the image
        img.close()

def modify_image(file_path):
    # Open the image file
    with Image.open(file_path) as img:
        # Ensure the image is in RGBA mode
        img = Image.open(file_path)
        # Get the image size
        width, height = img.size
        # Load the image data
        pixels = img.load()

        # Iterate through each pixel
        for y in range(height):
            for x in range(width):
                r, g, b = pixels[x, y]
                if  g >= 210 and r<50 and b<50:
                    # Change to r=0, g=255, b=0, keeping a same as before
                    pixels[x, y] = (0, 255, 0)

                elif r >=210 and g<50 and b<50:
                    # Change to r=0, g=255, b=0, keeping a same as before
                    pixels[x, y] = ( 255,0, 0)
                else:
                    # Change to r=0, g=0, b=255, keeping a same as before
                    pixels[x, y] = (0, 0, 255)

        # Save the modified image back to the same file
        img.save(file_path)

def modify_images_in_directory(directory):
    for filename in os.listdir(directory):
        if filename.lower().endswith('.png'):
            file_path = os.path.join(directory, filename)
            # print(f"Modifying {file_path}")
            modify_image(file_path)

# Specify the path to your directory
directory = segment_path

# Call the function to modify all PNG images in the directory
modify_images_in_directory(directory)

# Parameters
weed_color = (0, 0, 255)   # Red color (weed)
crop_color = (0, 255, 0)   # Green color (crop)
background_color = (255, 0, 0)  # Blue color (background)
yellow_color = (0, 255, 255)  # Yellow color for testing placement
radius_cm = 5   # Radius of circular weed-removal object in cm
pixel_size_cm = 0.1  # Assume each pixel represents 0.375 cm
r_pixels = int(radius_cm / pixel_size_cm)

# Paths
result_path = os.path.join(HOME, 'output')


# Ensure result_path directory exists
os.makedirs(result_path, exist_ok=True)

# Helper functions
def is_color(pixel, color):
    return (pixel == color).all()

def find_weed_regions(image):
    """Find all connected red regions in the image."""
    weed_mask = np.all(image == weed_color, axis=-1).astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(weed_mask)
    weed_regions = []
    for label in range(1, num_labels):
        region = np.argwhere(labels == label)
        weed_regions.append(region)
    return weed_regions

def can_place_circle(x, y, r_pixels, image):
    """Check if a circle centered at (x, y) with radius r_pixels can be placed without overlapping green pixels."""
    height, width, _ = image.shape
    for i in range(-r_pixels, r_pixels + 1):
        for j in range(-r_pixels, r_pixels + 1):
            if i**2 + j**2 <= r_pixels**2:
                nx, ny = x + i, y + j
                if 0 <= nx < width and 0 <= ny < height:
                    if is_color(image[ny, nx], crop_color):
                        return False
    return True

def get_weed_removal_positions(image, weed_regions):
    """Identify positions to place weed-removal objects for each weed region."""
    removal_positions = []

    for region in weed_regions:
        min_y, min_x = np.min(region, axis=0)
        max_y, max_x = np.max(region, axis=0)

        # Adjust the grid step to increase density of circles
        for y in range(min_y, max_y + 1, int(r_pixels * 1.32)):  # Reduced step size for more coverage
            for x in range(min_x, max_x + 1, int(r_pixels * 1.32)):
                if any((pt == [y, x]).all() for pt in region):
                    if can_place_circle(x, y, r_pixels, image):
                        removal_positions.append((x, y))
                        cv2.circle(image, (x, y), r_pixels, yellow_color, -1)
    return removal_positions

# Process each image in segment_path
res={}
for filename in os.listdir(segment_path):
    image_path = os.path.join(segment_path, filename)
    image = cv2.imread(image_path)

    if image is None:
        print(f"Warning: Could not load image {filename}. Skipping...")
        continue

    # Process the image
    weed_regions = find_weed_regions(image)
    removal_positions = get_weed_removal_positions(image, weed_regions)
    res[filename]=removal_positions

    # Save the output image with marked weed-removal areas
    output_path = os.path.join(result_path, filename)
    cv2.imwrite(output_path, image)
    # print(f"Saved annotated image to {output_path}")


filename = os.path.join(HOME, "res.json")

ress = {"x": 500, "y": 1105, "z": 0}

# Write the data to a JSON file
with open(filename, 'w') as file:
    json.dump(res, file, indent=4)
