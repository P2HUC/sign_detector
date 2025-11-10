import os
import pickle

import mediapipe as mp
import cv2
import matplotlib.pyplot as plt


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3, max_num_hands=2)

DATA_DIR = './data'

data = []
labels = []
for dir_ in os.listdir(DATA_DIR):
    dir_path = os.path.join(DATA_DIR, dir_)
    if not os.path.isdir(dir_path):
        # Skip files like .gitignore
        continue
    for img_path in os.listdir(dir_path):
        img_full_path = os.path.join(dir_path, img_path)
        if not os.path.isfile(img_full_path):
            continue
        # Process only common image extensions
        if not img_full_path.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            continue
        data_aux = []

        x_ = []
        y_ = []

        img = cv2.imread(img_full_path)
        if img is None:
            # Unreadable image; skip
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            # Process up to 2 hands
            hands_to_process = results.multi_hand_landmarks[:2]
            for hand_landmarks in hands_to_process:
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

            if data_aux:
                data.append(data_aux)
                labels.append(dir_)

with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)
