import pickle

import cv2
import mediapipe as mp
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

def open_first_available_camera(preferred_indices=(0, 1, 2, 3)):
    for idx in preferred_indices:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                return cap
            cap.release()
    return None

cap = open_first_available_camera()
if cap is None:
    print("ERROR: No available camera. Connect/enable a webcam and rerun.")
    raise SystemExit(1)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3, max_num_hands=2)


def find_font():
    """Try to locate a TrueType font that supports Vietnamese characters.

    Priority:
    1. Environment variable `SIGN_FONT_PATH`
    2. Common Windows fonts in `C:\\Windows\\Fonts`
    3. Fall back to PIL default font (may not support Vietnamese)
    """
    env_path = os.environ.get('SIGN_FONT_PATH')
    if env_path and os.path.isfile(env_path):
        return env_path

    # Local fonts dir (repo) - prefer a bundled unicode font
    fonts_dir = os.path.join(os.path.dirname(__file__), 'fonts')
    os.makedirs(fonts_dir, exist_ok=True)
    noto_path = os.path.join(fonts_dir, 'NotoSansVN-Regular.ttf')
    if os.path.isfile(noto_path):
        return noto_path

    # Common Windows font candidates
    candidates = [
        r'C:\Windows\Fonts\ARIALUNI.TTF',
        r'C:\Windows\Fonts\arial.ttf',
        r'C:\Windows\Fonts\times.ttf',
        r'C:\Windows\Fonts\seguiemj.ttf',
        r'C:\Windows\Fonts\tahoma.ttf',
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p

    # Try downloading Noto Sans VN (open-source) into ./fonts/
    try:
        import urllib.request
        font_url = 'https://raw.githubusercontent.com/googlefonts/noto-fonts/main/hinted/ttf/NotoSansVN/NotoSansVN-Regular.ttf'
        urllib.request.urlretrieve(font_url, noto_path)
        return noto_path
    except Exception:
        return None


def draw_text_pil(img_bgr, text, position, font_path=None, font_size=32, color=(0, 0, 0), outline=(255, 255, 255), stroke_width=2):
    """Draw Unicode text onto an OpenCV BGR image using Pillow and return BGR image.

    - img_bgr: OpenCV image in BGR
    - text: string (can be Unicode)
    - position: (x, y) top-left
    - font_path: optional TrueType font path
    - color: text color as BGR tuple
    - outline: outline color as BGR tuple
    - stroke_width: outline thickness
    """
    # Convert BGR to RGB for Pillow
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)

    # Determine font
    font = None
    if font_path:
        try:
            font = ImageFont.truetype(font_path, font_size)
        except Exception:
            font = None
    if font is None:
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

    # Pillow expects color in RGB
    color_rgb = (color[2], color[1], color[0])
    outline_rgb = (outline[2], outline[1], outline[0])

    # Use stroke_width/stroke_fill (Pillow >=5.2 supports stroke)
    try:
        draw.text(position, text, font=font, fill=color_rgb, stroke_width=stroke_width, stroke_fill=outline_rgb)
    except TypeError:
        # Older Pillow: emulate outline by drawing text multiple times
        x, y = position
        for ox in range(-stroke_width, stroke_width + 1):
            for oy in range(-stroke_width, stroke_width + 1):
                draw.text((x + ox, y + oy), text, font=font, fill=outline_rgb)
        draw.text(position, text, font=font, fill=color_rgb)

    # Convert back to BGR OpenCV image
    result = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return result

# Load the label mapping from the training data
with open('data.pickle', 'rb') as f:
    data_dict = pickle.load(f)
    unique_labels = sorted(list(set(data_dict['labels'])))
    labels_dict = {i: label for i, label in enumerate(unique_labels)}
while True:

    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    if not ret or frame is None:
        if cv2.waitKey(10) == ord('q'):
            break
        continue

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        # Process up to 2 hands
        hands_to_process = results.multi_hand_landmarks[:2]
        for hand_landmarks in hands_to_process:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style(),
            )

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))
        
        # If only 1 hand detected, pad with zeros to maintain consistent feature size
        if len(hands_to_process) == 1:
            for i in range(21):  # 21 landmarks per hand
                data_aux.extend([0, 0])  # Add zero x,y coordinates

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10

        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        if len(data_aux) != 84:
            # Skip prediction if features do not match model expectation
            cv2.imshow('frame', frame)
            cv2.waitKey(1)
            continue
        prediction = model.predict([np.asarray(data_aux)])
        
        # Get the predicted label directly (it's already a string)
        predicted_character = prediction[0]

        frame = draw_text_pil(frame, str(predicted_character), (x1, max(0, y1 - 40)), font_path=find_font(), font_size=36, color=(0, 0, 0), outline=(255, 255, 255), stroke_width=3)

    # Add instruction text
    frame = draw_text_pil(frame, 'Press Q to quit', (10, 30), font_path=find_font(), font_size=18, color=(0, 0, 255), outline=(255, 255, 255), stroke_width=1)
    
    cv2.imshow('Sign Language Detector', frame)
    
    # Check for 'q' key press
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("Exiting...")
        break


print("Cleaning up resources...")
cap.release()
cv2.destroyAllWindows()
print("Done!")
