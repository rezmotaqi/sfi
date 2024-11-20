import pickle

import cv2
import numpy as np
from PIL import Image
from keras.models import load_model
from mtcnn import MTCNN


# Function to extract faces from the frame
def extract_face(frame, required_size=(160, 160)):
    detector = MTCNN()
    results = detector.detect_faces(frame)
    if results:
        x1, y1, width, height = results[0]['box']
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        face = frame[y1:y2, x1:x2]
        face_image = Image.fromarray(face)
        face_image = face_image.resize(required_size)
        face_array = np.asarray(face_image)
        return face_array
    return None


# Function to get embeddings from a face using the FaceNet model
def get_embedding(model, face_pixels):
    # Standardize pixel values
    face_pixels = face_pixels.astype('float32')
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    samples = np.expand_dims(face_pixels, axis=0)
    # Get the embedding
    yhat = model.predict(samples)
    return yhat[0]


# Load the pre-trained FaceNet model
facenet_model = load_model(
    'facenet_keras.h5')  # Make sure the model file exists

# Load the trained SVM model from the pickle file
with open('mohamad.pkl', 'rb') as f:
    content = pickle.load(f)

# Extract the classifier from the dictionary
svm_model = content['classifier']

# Optional: Extract other elements, like target embeddings, if needed
target_embeddings = content.get('target_embeddings')  # (Optional)

# Initialize video capture
video_capture = cv2.VideoCapture(
    "http://192.168.186.19:8080/video")  # Replace with your mobile camera stream URL

if not video_capture.isOpened():
    print("Error: Unable to open video stream.")
    exit()

print("Starting video stream... Press 'q' to exit.")

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to retrieve frame. Exiting...")
        break
    
    # Extract face from the frame
    face_pixels = extract_face(frame)
    if face_pixels is not None:
        # Get embedding from the face
        embedding = get_embedding(facenet_model, face_pixels)
        embedding = embedding.reshape(1, -1)  # Reshape for SVM input
        
        # Perform prediction
        prediction = svm_model.predict(embedding)
        probability = svm_model.predict_proba(embedding)[0][1]
        
        # Define label and color
        if prediction == 1 and probability > 0.5:
            label = "Mohamad"
            color = (0, 255, 0)  # Green for known faces
        else:
            label = "Unknown"
            color = (0, 0, 255)  # Red for unknown faces
        
        # Display label on the frame
        cv2.putText(frame, f"{label} ({probability:.2f})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    # Display the video feed
    cv2.imshow('Mobile Camera Feed', frame)
    
    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
video_capture.release()
cv2.destroyAllWindows()
