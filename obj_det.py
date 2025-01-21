## In cmd with image source
# yolo detect predict model=yolo11n.pt source='images/demo.png'

from ultralytics import YOLO

# Load a model
model = YOLO("models/yolov8s.pt")

# Train the model
train_results = model.train(
    data="coco8.yaml",
    epochs=30,  # number of training epochs
    imgsz=640,  # training image size
    device="cpu",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
)

# Evaluate model performance on the validation set
metrics = model.val()

# Perform object detection on an image
results = model("images/crowd3.jpg")
results[0].show()

# Export the model to ONNX format
path = model.export(format="onnx")  # return path to exported model
