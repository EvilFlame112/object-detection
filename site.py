import streamlit as st
import os
import subprocess
from pathlib import Path

# Set up directories
BASE_DIR = Path(__file__).resolve().parent
OBJ_DET_DIR = BASE_DIR / "obj_det_outputs"
RETINA_NET_DIR = BASE_DIR / "retina_net_outputs"
TRACKING_DIR = BASE_DIR / "tracking_outputs"

# Ensure output directories exist
OBJ_DET_DIR.mkdir(exist_ok=True)
RETINA_NET_DIR.mkdir(exist_ok=True)
TRACKING_DIR.mkdir(exist_ok=True)

# Helper function to execute scripts and save output
def execute_script(script_name, media_file, output_dir):
    output_path = output_dir / media_file.name
    command = ["python", script_name, str(media_file), str(output_path)]
    try:
        subprocess.run(command, check=True)
        return output_path
    except subprocess.CalledProcessError as e:
        st.error(f"Error executing {script_name}: {e}")
        return None

# Streamlit UI
def main():
    st.title("Object Detection and Tracking Platform")
    
    mode = st.sidebar.selectbox("Select Mode", ["Object Detection", "Retina Net", "Tracking"])

    uploaded_file = st.file_uploader("Upload your media file (image/video)", type=["jpg", "png", "mp4"])

    if uploaded_file is not None:
        st.write(f"**Uploaded file:** {uploaded_file.name}")
        
        if st.button("Run Analysis"):
            if mode == "Object Detection":
                output = execute_script("obj_det.py", uploaded_file, OBJ_DET_DIR)
            elif mode == "Retina Net":
                output = execute_script("retina_net.py", uploaded_file, RETINA_NET_DIR)
            elif mode == "Tracking":
                output = execute_script("tracking.py", uploaded_file, TRACKING_DIR)
            
            if output:
                st.success(f"Analysis complete. Output saved to: {output}")
                if output.suffix in [".jpg", ".png"]:
                    st.image(str(output))
                elif output.suffix == ".mp4":
                    st.video(str(output))

if __name__ == "__main__":
    main()
