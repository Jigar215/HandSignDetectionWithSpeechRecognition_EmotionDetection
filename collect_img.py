import os
import cv2
import string
import time

# Dataset configuration
DATA_DIR = './data'
number_of_classes = 26  # A-Z
dataset_size = 100  # Images per letter
labels = list(string.ascii_uppercase)  # ['A', 'B', ..., 'Z']

# Create base data directory
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Initialize webcam
cap = cv2.VideoCapture(0)

for j in range(number_of_classes):
    label = labels[j]
    class_dir = os.path.join(DATA_DIR, label)

    # Create directory for this letter
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print(f'📝 Get ready to collect data for letter: "{label}"')

    # Show a prompt and wait for 'q' to start or 'Esc' to exit
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        cv2.putText(frame, f'Prepare "{label}" - Press "Q" to start | Esc to exit', (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow('frame', frame)

        key = cv2.waitKey(25) & 0xFF
        if key == ord('q'):
            break
        elif key == 27:  # Esc key
            print("🚪 Exiting...")
            cap.release()
            cv2.destroyAllWindows()
            exit()

    print(f'📸 Capturing images for "{label}"...')
    time.sleep(2)  # Small delay before starting capture

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            continue

        cv2.putText(frame, f'Letter: {label} | Image: {counter + 1}/{dataset_size} | Esc to exit', (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)
        cv2.imshow('frame', frame)

        # Save image
        image_path = os.path.join(class_dir, f'{counter}.jpg')
        cv2.imwrite(image_path, frame)

        counter += 1
        key = cv2.waitKey(50) & 0xFF
        if key == 27:  # Esc to exit
            print("🚪 Exiting...")
            cap.release()
            cv2.destroyAllWindows()
            exit()

print("✅ Dataset collection complete!")

cap.release()
cv2.destroyAllWindows()
