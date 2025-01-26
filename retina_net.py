from imageai.Detection import VideoObjectDetection
import os
import time
import sys
from pathlib import Path

if __name__ == "__main__":
    # Get input and output paths from command line arguments
    input_file_path = sys.argv[1]
    output_file_path = sys.argv[2]

    output_file_path = str(Path(output_file_path).with_suffix(""))

    # Start timing
    start_time = time.time()

    execution_path = os.getcwd()

    detector = VideoObjectDetection()
    detector.setModelTypeAsTinyYOLOv3()
    detector.setModelPath(os.path.join(execution_path , "models/tiny-yolov3.pt"))
    detector.loadModel()

    detector.detectObjectsFromVideo(
        input_file_path=input_file_path,
        output_file_path=output_file_path,
        frames_per_second=10,
        log_progress=True
    )

    # End timing and calculate the duration
    end_time = time.time()
    execution_duration = end_time - start_time

    print(f"Video saved at: {output_file_path}")
    print("Time taken to run the code:", execution_duration, "seconds")
