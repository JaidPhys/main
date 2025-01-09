import streamlit as st
import cv2 as cv
import tempfile
import mediapipe as mp
import time
import asyncio
import firebase_admin
from firebase_admin import credentials, firestore

# Initialize MediaPipe Pose with real-time settings
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

st.title("Real-Time Pose Estimation on Video")

# File uploader for video
uploaded_file = st.file_uploader("Upload a video of a person", type=["mp4", "avi", "mov"])

show_feedback = False
if uploaded_file is not None:
    # Save uploaded video to a temporary file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    # Open video file using OpenCV
    vf = cv.VideoCapture(tfile.name)
    stframe = st.empty()  # Placeholder for displaying frames

    # Get video FPS
    fps = vf.get(cv.CAP_PROP_FPS)
    frame_time = 1 / fps if fps > 0 else 0.03  # Time between frames (original speed)
    slow_factor = 1.2  # Adjust this value for slightly slower playback (1.2 = 20% slower)
    delay = frame_time * slow_factor
    total_frames = int(vf.get(cv.CAP_PROP_FRAME_COUNT))

    st.success(f"fps: {round(fps, 1)}; Total frames in video: {total_frames}")

    while vf.isOpened():
        ret, frame = vf.read()
        if not ret:
            st.success(f"Analysis Complete. Total frames processed: {total_frames}")
            break

        # Resize frame for faster processing (optional)
        frame = cv.resize(frame, (640, 480))

        # Convert to RGB for MediaPipe
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        # Real-time pose processing
        results = pose.process(rgb_frame)

        # Draw pose landmarks if detected
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

        # Display the current frame with pose landmarks
        stframe.image(frame, channels="BGR", use_container_width=True)

        # Introduce a small delay for slightly slower playback
        time.sleep(delay)

    vf.release()
    show_feedback = True


# --- Firebase Feedback Section ---
async def get_feedback():
    try:
        # Load the TOML content from an environment variable
        credentials_data = st.secrets.FIREBASE_SERVICE_ACCOUNT_KEY
    except Exception as e:
        st.error(f"Errore nella lettura delle credenziali: {e}")
        return

    try:
        cred = credentials.Certificate({
            "type": credentials_data.type,
            "project_id": credentials_data.project_id,
            "private_key_id": credentials_data.private_key_id,
            "private_key": credentials_data.private_key.replace("\\n", "\n"),
            "client_email": credentials_data.client_email,
            "client_id": credentials_data.client_id,
            "auth_uri": credentials_data.auth_uri,
            "token_uri": credentials_data.token_uri,
            "auth_provider_x509_cert_url": credentials_data.auth_provider_x509_cert_url,
            "client_x509_cert_url": credentials_data.client_x509_cert_url,
            "universe_domain": credentials_data.universe_domain
        })
    except Exception as e:
        st.error(f"Errore nella conversione delle credenziali: {e}")
        return

    try:
        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred)
    except Exception as e:
        st.error(f"Errore nello stabilire una connessione: {e}")
        return

    try:
        db = firestore.client()
        feedback_ref = db.collection('feedback').order_by('timestamp', direction=firestore.Query.DESCENDING).limit(1)
    except Exception as e:
        st.error(f"Errore nel recupero del feedback: {e}")
        return

    try:
        feedback = feedback_ref.stream()
        if feedback:
            for fb in feedback:
                return fb.to_dict()['content']
        else:
            return "Nessun feedback disponibile."
    except Exception as e:
        st.error(f"Errore nella scrittura del feedback: {e}")


if show_feedback:
    feedback = asyncio.run(get_feedback())
    if feedback:
        st.text_area("Feedback", value=feedback, height=300, disabled=True)
