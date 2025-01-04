from imageai.Detection import VideoObjectDetection
import os
import time

# Start timing
start_time = time.time()

execution_path = os.getcwd()

detector = VideoObjectDetection()
# detector.useCPU() #to use CPU --> by default it uses GPU

# #to use Retinanet model
# detector.setModelTypeAsRetinaNet()

# #to use Yolov3 model
# detector.setModelTypeAsYOLOv3()

#to use TinyYolo model
detector.setModelTypeAsTinyYOLOv3()

detector.setModelPath( os.path.join(execution_path , "models/tiny-yolov3.pt"))
detector.loadModel()

video_path = detector.detectObjectsFromVideo(input_file_path=os.path.join(execution_path, "videos/00abd8a7-ecd6fc56.mov"),
                            output_file_path=os.path.join(execution_path, "traffic_detected_retinanet")
                            , frames_per_second=10, log_progress=True)
print(video_path)

# End timing and calculate the duration
end_time = time.time()
execution_duration = end_time - start_time

# print("Video saved at:", output_file_path)
print("Time taken to run the code:", execution_duration, "seconds")