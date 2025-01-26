import streamlit as st
import os
import subprocess
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploaded_files"
OBJ_DET_DIR = BASE_DIR / "obj_det_outputs"
OBJECT_DETECTION_DIR = BASE_DIR / "object_detection_outputs"
RETINA_NET_DIR = BASE_DIR / "retina_net_outputs"
TRACKING_DIR = BASE_DIR / "tracking_outputs"

UPLOAD_DIR.mkdir(exist_ok=True)
OBJ_DET_DIR.mkdir(exist_ok=True)
OBJECT_DETECTION_DIR.mkdir(exist_ok=True)
RETINA_NET_DIR.mkdir(exist_ok=True)
TRACKING_DIR.mkdir(exist_ok=True)

def save_uploaded_file(uploaded_file):
    file_path = UPLOAD_DIR / uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def execute_script(script_name, media_file_path, output_dir):
    if script_name == "tracking.py":
        # For tracking.py, always use .mp4 extension
        output_path = output_dir / f"{media_file_path.stem}.mp4"
    else:
        output_path = output_dir / media_file_path.name

    python_interpreter = Path(sys.executable) 
    
    command = [str(python_interpreter), script_name, str(media_file_path), str(output_path)]
    try:
        subprocess.run(command, check=True)
        return output_path
    except subprocess.CalledProcessError as e:
        st.error(f"Error executing {script_name}: {e}")
        return None

def delete_uploaded_file(file_path):
    """Delete the uploaded file after processing."""
    if file_path.exists():
        file_path.unlink()
        print(f"Deleted uploaded file: {file_path}")
    else:
        print(f"File not found: {file_path}")

def main():
    st.set_page_config(page_title="Object Detection and Tracking Platform", page_icon=":camera:")
    st.title("Object Detection and Tracking Platform")
    
    mode = st.sidebar.selectbox("Select Mode", ["Object Detection", "Object Detection - Video", "Retina Net", "Tracking"])

    uploaded_file = st.file_uploader("Upload your media file (image/video)", type=["jpg", "png", "mp4", "avi", "mov"])

    if uploaded_file is not None:
        st.write(f"**Uploaded file:** {uploaded_file.name}")
        file_path = save_uploaded_file(uploaded_file)
        
        if st.button("Run Analysis"):
            if mode == "Object Detection":
                output = execute_script("obj_det.py", file_path, OBJ_DET_DIR)
            if mode == "Object Detection - Video":
                output = execute_script("od_yolo_tiny_w.py", file_path, OBJECT_DETECTION_DIR)
            elif mode == "Retina Net":
                output = execute_script("retina_net.py", file_path, RETINA_NET_DIR)
            elif mode == "Tracking":
                output = execute_script("tracking.py", file_path, TRACKING_DIR)
            
            if output:
                st.success(f"Analysis complete. Output saved to: {output}")
                if output.suffix in [".jpg", ".png"]:
                    st.image(str(output))
                elif output.suffix == ".mp4" or output.suffix == ".avi":
                    st.video(str(output))
                delete_uploaded_file(file_path)

if __name__ == "__main__":
    main()
