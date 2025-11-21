import streamlit as st
try:
    import cv2
    cv2_import_error = None
except Exception as e:
    cv2 = None
    cv2_import_error = e
import numpy as np
import mediapipe as mp
import pickle
from PIL import Image, ImageDraw, ImageFont
import av
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode, RTCConfiguration
import os

# Set page config
st.set_page_config(
    page_title="Sign Language Detection",
    page_icon="👋",
    layout="wide"
)

# Title and description
st.title("Sign Language Detection")
st.write("Real-time sign language detection using MediaPipe and machine learning")

    # Load the model
try:
    model_dict = pickle.load(open('./model.p', 'rb'))
    model = model_dict['model']
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    model = None

    # If cv2 failed to import, show a helpful error and stop
    if cv2 is None:
        st.error("❌ OpenCV failed to import on this environment.\n"
                 "This commonly happens on Streamlit Cloud where OpenCV GUI binaries are not available.\n"
                 "For deployment, ensure `requirements.txt` uses `opencv-python-headless`.\n"
                 f"Import error details: {cv2_import_error}")
        st.stop()

# RTC Configuration for WebRTC
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Video Transformer Class
class SignLanguageDetector(VideoTransformerBase):
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        # For video, use static_image_mode=False for better performance
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            min_detection_confidence=0.5,
            max_num_hands=2
        )
        self.font_path = self.find_font()

    def find_font(self):
        env_path = os.environ.get('SIGN_FONT_PATH')
        if env_path and os.path.isfile(env_path):
            return env_path
        candidates = [
            r'C:\Windows\Fonts\ARIALUNI.TTF',
            r'C:\Windows\Fonts\arial.ttf',
            r'C:\Windows\Fonts\tahoma.ttf',
        ]
        for p in candidates:
            if os.path.isfile(p):
                return p
        return None

    def draw_text_pil(self, img_bgr, text, position, font_path=None, font_size=24, color=(0,0,0), outline=(255,255,255), stroke_width=2):
        # Convert to RGB and use Pillow to draw Unicode text
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(pil_img)
        font = None
        if font_path:
            try:
                font = ImageFont.truetype(font_path, font_size)
            except Exception:
                font = ImageFont.load_default()
        else:
            font = ImageFont.load_default()
        try:
            draw.text(position, text, font=font, fill=(color[2], color[1], color[0]), stroke_width=stroke_width, stroke_fill=(outline[2], outline[1], outline[0]))
        except TypeError:
            x, y = position
            for ox in range(-stroke_width, stroke_width+1):
                for oy in range(-stroke_width, stroke_width+1):
                    draw.text((x+ox, y+oy), text, font=font, fill=(outline[2], outline[1], outline[0]))
            draw.text(position, text, font=font, fill=(color[2], color[1], color[0]))
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        
    def recv(self, frame):
        try:
            img = frame.to_ndarray(format="bgr24")
            # Resize incoming frames to a fixed width to reduce processing cost
            target_w = 640
            h, w, _ = img.shape
            if w != target_w:
                scale = target_w / float(w)
                img = cv2.resize(img, (target_w, int(h * scale)))
            H, W, _ = img.shape
            
            # Process the image
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.hands.process(img_rgb)
            
            if results.multi_hand_landmarks:
                data_aux = []
                x_ = []
                y_ = []
                
                # Process up to 2 hands
                hands_to_process = results.multi_hand_landmarks[:2]
                
                for hand_landmarks in hands_to_process:
                    # Draw hand landmarks
                    mp.solutions.drawing_utils.draw_landmarks(
                        img,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                        mp.solutions.drawing_styles.get_default_hand_connections_style()
                    )
                    
                    # Collect x, y coordinates
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        x_.append(x)
                        y_.append(y)
                    
                    # Prepare data for prediction (relative coordinates)
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        data_aux.append(x - min(x_))
                        data_aux.append(y - min(y_))
                
                # If only 1 hand detected, pad with zeros
                if len(hands_to_process) == 1:
                    for _ in range(21):  # 21 landmarks per hand
                        data_aux.extend([0, 0])
                
                # Make prediction if we have the right number of features
                if model is not None and len(data_aux) == 84:  # 42 points * 2 (x,y)
                    prediction = model.predict([np.asarray(data_aux)])
                    predicted_character = prediction[0]  # Directly use the predicted character
                    
                    # Draw bounding box and prediction
                    x1 = int(min(x_) * W) - 10
                    y1 = int(min(y_) * H) - 10
                    x2 = int(max(x_) * W) + 10
                    y2 = int(max(y_) * H) + 10
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 4)
                    # Use Pillow to render Unicode (Vietnamese) text
                    if self.font_path:
                        img = self.draw_text_pil(img, str(predicted_character), (x1, max(0, y1 - 40)), font_path=self.font_path, font_size=32, color=(0,0,255), outline=(255,255,255), stroke_width=2)
                    else:
                        cv2.putText(img, predicted_character, (x1, y1 - 10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3, 
                                  cv2.LINE_AA)
            
            return av.VideoFrame.from_ndarray(img, format="bgr24")
            
        except Exception as e:
            st.error(f"Error in video processing: {e}")
            return frame

# Main App
def main():
    st.sidebar.title("Settings")
    
    # Instructions
    st.sidebar.markdown("### How to use:")
    st.sidebar.markdown("1. Click 'Start' to begin the webcam")
    st.sidebar.markdown("2. Show hand signs to the camera")
    st.sidebar.markdown("3. The detected sign will be displayed on the screen")
    
    # Model info
    if model is not None:
        st.sidebar.markdown("### Model Status")
        st.sidebar.success("✅ Model is ready")
        if hasattr(model, 'classes_'):
            st.sidebar.write(f"Number of classes: {len(model.classes_)}")
        st.sidebar.markdown("### How to use")
        st.sidebar.write("1. Click 'Start' to begin")
        st.sidebar.write("2. Show hand signs to the camera")
    
    # WebRTC Streamer
    st.markdown("### Live Detection")
    # Request a higher resolution camera if available; browsers will honor if supported
    media_constraints = {
        "video": {"width": {"ideal": 1280}, "height": {"ideal": 720}},
        "audio": False,
    }

    webrtc_ctx = webrtc_streamer(
        key="sign-language-detection",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=SignLanguageDetector,
        media_stream_constraints=media_constraints,
        async_processing=True,
    )

    # Performance tips
    st.sidebar.markdown("**Performance tips:**")
    st.sidebar.markdown("- Use Chrome/Edge for best WebRTC support")
    st.sidebar.markdown("- Allow the camera to use a 720p or 1080p stream when prompted")
    st.sidebar.markdown("- Close other camera-using apps to increase frame-rate")

if __name__ == "__main__":
    if model is None:
        st.error("❌ Failed to load the model. Please check if model.p exists and is valid.")
    else:
        main()
