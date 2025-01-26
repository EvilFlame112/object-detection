import cv2
import numpy as np
from object_detection import ObjectDetection
import math
import sys
from pathlib import Path
import subprocess

def main(input_video_path, output_video_path):
    # Initialize Object Detection
    od = ObjectDetection()

    # Open the input video
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {input_video_path}")
        return None

    # Get video properties for saving
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # Use CAP_PROP_FPS instead of CAP_PROP_FRAME_RATE

    # Ensure the output file has a .mp4 extension
    output_video_path = output_video_path.with_suffix(".mp4")

    # Create the output video writer
    output_video = cv2.VideoWriter(
        str(output_video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height)
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

        # Prepare for next frame
        center_points_prev_frame = center_points_cur_frame.copy()

    # Release resources
    cap.release()
    output_video.release()

    # Re-encode the video to ensure compatibility with Streamlit
    reencoded_output_path = output_video_path.with_stem(output_video_path.stem + "_reencoded")
    ffmpeg_command = [
        "ffmpeg", "-i", str(output_video_path),
        "-vcodec", "libx264", "-acodec", "aac",
        str(reencoded_output_path)
    ]

    try:
        subprocess.run(ffmpeg_command, check=True)
        print(f"Video re-encoded and saved at: {reencoded_output_path}")
        # Replace the original output with the re-encoded version
        output_video_path.unlink()  # Delete the original file
        reencoded_output_path.rename(output_video_path)  # Rename the re-encoded file
    except subprocess.CalledProcessError as e:
        print(f"Error re-encoding video: {e}")
        # If re-encoding fails, use the original output file
        reencoded_output_path = output_video_path

    return output_video_path

if __name__ == "__main__":
    # Get input and output paths from command line arguments
    input_video_path = sys.argv[1]
    output_video_path = Path(sys.argv[2])

    # Run the tracking process
    output_path = main(input_video_path, output_video_path)
    if output_path:
        print(f"Tracking complete. Output saved to: {output_path}")