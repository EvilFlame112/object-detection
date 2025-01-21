import cv2
import numpy as np
from object_detection import ObjectDetection
import math

# Initialize Object Detection
od = ObjectDetection()

cap = cv2.VideoCapture("videos/video.mp4")

# Get video properties for saving
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
output_video = cv2.VideoWriter(
    "output/tracking_obj_video.avi", cv2.VideoWriter_fourcc(*"XVID"), fps, (frame_width, frame_height)
)

# Initialize variables
count = 0
center_points_prev_frame = []
tracking_objects = {}
track_id = 0

while True:
    ret, frame = cap.read()
    count += 1
    if not ret:
        break

    # Store center points of the current frame
    center_points_cur_frame = []

    # Detect Objects on the frame
    (class_ids, scores, boxes) = od.detect(frame)
    for box in boxes:
        (x, y, w, h) = box

        # Calculate center points of the bounding box
        cx = int((x + x + w) / 2)
        cy = int((y + y + h) / 2)
        center_points_cur_frame.append((cx, cy))

        # Draw rectangle around detected object
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Update tracking objects
    for pt in center_points_cur_frame:
        same_object_detected = False
        for object_id, prev_pt in tracking_objects.items():
            distance = math.hypot(prev_pt[0] - pt[0], prev_pt[1] - pt[1])

            if distance < 35:  # Threshold distance
                tracking_objects[object_id] = pt
                same_object_detected = True
                break

        # Assign new ID to new object
        if not same_object_detected:
            tracking_objects[track_id] = pt
            track_id += 1

    # Draw tracking points and IDs
    for object_id, pt in tracking_objects.items():
        # Draw a filled circle for tracking
        cv2.circle(frame, pt, 5, (0, 0, 255), -1)

        # Ensure numbers are drawn last, on top of all objects
        cv2.putText(
            frame,
            str(object_id),
            (pt[0] - 10, pt[1] - 10),  # Offset for better visibility
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),  # White text for visibility
            2,  # Thickness
            lineType=cv2.LINE_AA,  # Anti-aliased text
        )

    # Write the processed frame to the output video
    output_video.write(frame)

    # Show the full frame
    # cv2.imshow("Full Frame with Detection", frame)

    # Prepare for next frame
    center_points_prev_frame = center_points_cur_frame.copy()

    # Press ESC to exit
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
output_video.release()
cv2.destroyAllWindows()
