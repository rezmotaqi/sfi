import os
import time
import pickle
import cv2
import face_recognition
import numpy as np
import pygame

# Initialize Pygame mixer for audio playback
pygame.mixer.init()

# Load the audio file
audio_file = "alie.mp3"
if not os.path.isfile(audio_file):
    raise FileNotFoundError(f"Audio file '{audio_file}' not found.")
pygame.mixer.music.load(audio_file)

# Load the trained classifier and target embeddings
model_path = 'mohamad.pkl'
if not os.path.isfile(model_path):
    raise FileNotFoundError(f"Model file '{model_path}' not found.")
with open(model_path, 'rb') as f:
    model_data = pickle.load(f)
classifier = model_data['classifier']
target_embeddings = model_data['target_embeddings']

# Initialize webcam
video_capture = cv2.VideoCapture(0)
if not video_capture.isOpened():
    raise Exception("Webcam not initialized properly.")

# Set the desired resolution (e.g., 1280x720)
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Variables to manage audio playback
target_detected = False
last_detection_time = 0
detection_timeout = 2  # seconds

print("Starting real-time face recognition. Press 'q' to quit.")

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to capture frame.")
        break

    # Convert the frame from BGR to RGB
    rgb_frame = np.ascontiguousarray(frame[:, :, ::-1])

    # Find all face locations and face encodings in the current frame
    face_locations = face_recognition.face_locations(rgb_frame, model="cnn")
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    current_time = time.time()
    target_currently_detected = False

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Reshape the encoding for prediction
        face_encoding_reshaped = face_encoding.reshape(1, -1)
        # Predict using the classifier
        prediction = classifier.predict(face_encoding_reshaped)
        probability = classifier.predict_proba(face_encoding_reshaped)[0][1]

        if prediction == 1 and probability > 0.5:  # Adjust threshold as needed
            target_currently_detected = True
            last_detection_time = current_time
            print(f"Ali hallaji is detected at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(current_time))}")

            # Draw rectangle around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            # Display the label below the rectangle
            cv2.putText(frame, "Ali Hallaji", (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        else:
            # Draw rectangle around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            # Display the label below the rectangle
            cv2.putText(frame, "Unknown", (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    # Manage audio playback
    if target_currently_detected and not target_detected:
        # Start playing audio if target is detected and audio is not already playing
        pygame.mixer.music.play(-1)  # -1 means loop indefinitely
        target_detected = True
    elif not target_currently_detected and target_detected:
        # Stop playing audio if target is no longer detected
        if current_time - last_detection_time > detection_timeout:
            pygame.mixer.music.stop()
            target_detected = False

    # Display the resulting frame
    cv2.imshow("Real-time Face Recognition", frame)

    # Quit with 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
video_capture.release()

