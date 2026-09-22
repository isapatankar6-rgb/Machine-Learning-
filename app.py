import streamlit as st
import cv2
import numpy as np
import mediapipe as mp

# Configure page UI
st.set_page_config(page_title="AI Pose Classifier", layout="wide")
st.title("🏋️‍♂️ AI Movement & Pose Classifier")
st.write("Upload an image to analyze body keypoints and movement phases.")

# Sidebar controls
st.sidebar.header("Model Configuration")
model_choice = st.sidebar.selectbox(
    "Select Model",
    ["Custom MLP (Module 6)", "SVM Baseline", "Decision Tree Baseline"]
)

# MediaPipe Pose Initializer
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# File Uploader
uploaded_file = st.file_uploader("Upload a pose image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Convert uploaded file to OpenCV format
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process image with MediaPipe
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            # Draw pose landmarks on frame
            annotated_image = image_rgb.copy()
            mp_drawing.draw_landmarks(
                annotated_image, 
                results.pose_landmarks, 
                mp_pose.POSE_CONNECTIONS
            )

            col1, col2 = st.sidebar.columns(2)
            st.image(annotated_image, caption="Detected Pose Landmarks", use_column_width=True)
            
            # Simulated model output
            st.success(f"**Selected Architecture:** {model_choice}")
            st.info("**Predicted Phase:** Concentric Phase / Lockout")
        else:
            st.error("No pose detected in the image. Please upload a clearer photo.")
