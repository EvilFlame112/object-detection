from ultralytics import YOLO
import sys
from pathlib import Path
from PIL import Image
import cv2

# Load a model
model = YOLO("models/yolov8s.pt")

# Check for input arguments
if len(sys.argv) != 3:
    print("Usage: python obj_det.py <input_image_path> <output_image_path>")
    sys.exit(1)

input_image_path = Path(sys.argv[1])
output_image_path = Path(sys.argv[2])

# Ensure the input file exists
if not input_image_path.exists():
    print(f"Error: Input file {input_image_path} does not exist.")
    sys.exit(1)

# Open the input image to ensure it is valid
try:
    with Image.open(input_image_path) as img:
        img.verify()
except Exception as e:
    print(f"Error: Cannot identify image file {input_image_path}. {e}")
    sys.exit(1)

# Perform object detection on the provided image
results = model(str(input_image_path))

# Save the results to the specified output path
output_image_path.parent.mkdir(parents=True, exist_ok=True)
results_img = results[0].plot()
cv2.imwrite(str(output_image_path), results_img)

print(f"Detection complete. Results saved to {output_image_path}")