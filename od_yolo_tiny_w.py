from imageai.Detection import VideoObjectDetection
import os
import time
import sys
from pathlib import Path

def forFrame(frame_number, output_array, output_count):
    print("FOR FRAME ", frame_number)
    print("Output for each object : ", output_array)
    print("Output count for unique objects : ", output_count)
    print("------------END OF A FRAME --------------")

def forSeconds(second_number, output_arrays, count_arrays, average_output_count):
    print("SECOND : ", second_number)
    print("Array for the outputs of each frame ", output_arrays)
    print("Array for output count for unique objects in each frame : ", count_arrays)
    print("Output average count for unique objects in the last second: ", average_output_count)
    print("------------END OF A SECOND --------------")

def forMinute(minute_number, output_arrays, count_arrays, average_output_count):
    print("MINUTE : ", minute_number)
    print("Array for the outputs of each frame ", output_arrays)
    print("Array for output count for unique objects in each frame : ", count_arrays)
    print("Output average count for unique objects in the last minute: ", average_output_count)
    print("------------END OF A MINUTE --------------")

def main(input_video_path, output_video_path):
    # Start timing
    start_time = time.time()

    # Set the execution path
    execution_path = os.getcwd()

    # Initialize the video detector
    video_detector = VideoObjectDetection()
    video_detector.setModelTypeAsTinyYOLOv3()
    video_detector.setModelPath(os.path.join(execution_path, "models/tiny-yolov3.pt"))
    video_detector.loadModel()

    # Ensure the output file has a .mp4 extension
    output_video_path = output_video_path.with_suffix(".mp4")
    output_video_path = str(Path(output_video_path).with_suffix(""))

    # Perform object detection on the video
    video_detector.detectObjectsFromVideo(
        input_file_path=input_video_path,
        output_file_path=str(output_video_path),
        frames_per_second=10,
        per_second_function=forSeconds,
        per_frame_function=forFrame,
        per_minute_function=forMinute,
        minimum_percentage_probability=30,
        log_progress=True
    )

    # End timing and calculate the duration
    end_time = time.time()
    execution_duration = end_time - start_time

    print(f"Video saved at: {output_video_path}")
    print("Time taken to run the code:", execution_duration, "seconds")

    return output_video_path

if __name__ == "__main__":
    # Get input and output paths from command line arguments
    input_video_path = sys.argv[1]
    output_video_path = Path(sys.argv[2])

    # Run the object detection process
    output_path = main(input_video_path, output_video_path)
    if output_path:
        print(f"Object detection complete. Output saved to: {output_path}")