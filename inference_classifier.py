import pickle

import cv2
import mediapipe as mp
import numpy as np

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

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                    cv2.LINE_AA)

    # Add instruction text
    cv2.putText(frame, 'Press Q to quit', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
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
