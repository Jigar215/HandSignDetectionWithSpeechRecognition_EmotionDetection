import os
import pickle
import mediapipe as mp
import cv2

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Dataset directory
DATA_DIR = './data'

data = []
labels = []

# Loop through each letter folder (e.g., A, B, ..., Z)
for dir_ in sorted(os.listdir(DATA_DIR)):
    class_path = os.path.join(DATA_DIR, dir_)
    if not os.path.isdir(class_path):
        continue  # Skip non-folder files if any

    print(f"Processing class: {dir_} ({len(os.listdir(class_path))} images)")
    
    for img_path in os.listdir(class_path):
        image_file = os.path.join(class_path, img_path)
        img = cv2.imread(image_file)
        if img is None:
            continue  # Skip unreadable files

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                data_aux = []
                x_, y_ = [], []

                # Collect x, y coordinates
                for lm in hand_landmarks.landmark:
                    x_.append(lm.x)
                    y_.append(lm.y)

                for lm in hand_landmarks.landmark:
                    data_aux.append(lm.x - min(x_))
                    data_aux.append(lm.y - min(y_))

                # Only accept frames with complete hand landmarks (21 points)
                if len(data_aux) == 42:
                    data.append(data_aux)
                    labels.append(dir_)

# Save dataset
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print(f"\n✅ Dataset saved as 'data.pickle' with {len(data)} samples.")
