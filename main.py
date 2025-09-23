import cv2
import mediapipe as mp
import pickle
import numpy as np
import pyttsx3
import threading
import time
from collections import deque

# Load the trained model
with open("model.p", "rb") as f:
    model_dict = pickle.load(f)
model = model_dict["model"]

# Load label map
label_map = model_dict.get("label_map", None)
if label_map is None:
    raise ValueError("Error: 'label_map' not found in model.p. Please include it when saving the model.")

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

# Text-to-speech setup with a dedicated thread
class TTSThread(threading.Thread):
    def __init__(self):
        super().__init__()
        self.queue = deque()
        self.lock = threading.Lock()
        self.running = True
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", 150)
        self.last_spoken = None
        self.last_spoken_time = 0
        
    def run(self):
        while self.running:
            if self.queue:
                with self.lock:
                    text = self.queue.popleft()
                
                # Only speak if it's a new word or enough time has passed
                current_time = time.time()
                if text != self.last_spoken or (current_time - self.last_spoken_time) > 2.0:
                    try:
                        self.engine.say(text)
                        self.engine.runAndWait()
                        self.last_spoken = text
                        self.last_spoken_time = current_time
                    except Exception as e:
                        print(f"TTS error: {e}")
            time.sleep(0.1)  # Small delay to prevent busy waiting
            
    def add_text(self, text):
        with self.lock:
            # Only add if not already in queue
            if text not in self.queue:
                self.queue.append(text)
                
    def stop(self):
        self.running = False

# Initialize TTS thread
tts_thread = TTSThread()
tts_thread.start()

# Initialize webcam
cap = cv2.VideoCapture(0)

prediction_buffer = []
STABILITY_THRESHOLD = 12  # more frames → more stable prediction

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame for natural webcam view
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            prediction = model.predict([landmarks])[0]
            predicted_character = label_map[prediction]

            # Add prediction to buffer
            prediction_buffer.append(predicted_character)
            if len(prediction_buffer) > STABILITY_THRESHOLD:
                prediction_buffer.pop(0)

            # Display current prediction
            cv2.putText(frame, f"Prediction: {predicted_character}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Get stable prediction
            if len(prediction_buffer) >= STABILITY_THRESHOLD:
                stable_prediction = max(set(prediction_buffer), key=prediction_buffer.count)
                
                # Display stable prediction
                cv2.putText(frame, f"Stable: {stable_prediction}", (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                
                # Send to TTS thread
                tts_thread.add_text(stable_prediction)
    else:
        prediction_buffer = []

    cv2.imshow("Sign Language Recognition", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

# Clean up
tts_thread.stop()
tts_thread.join()
cap.release()
cv2.destroyAllWindows()