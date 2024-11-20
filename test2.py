import pickle
import numpy as np
import cv2
from keras.models import load_model
from mtcnn import MTCNN
from PIL import Image

# Load the FaceNet model
facenet_model = load_model('facenet_keras.h5')

# Load the pre-trained SVM model
with open('mohamad.pkl', 'rb') as f:
    content = pickle.load(f)
svm_model = content['classifier']
target_embeddings = content.get('target_embeddings')  # Optional, for testing SVM accuracy

# Initialize MTCNN for face detection
detector = MTCNN()

# Function to preprocess face images
def preprocess_face(face_pixels):
    face_pixels = face_pixels.astype('float32')
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    return face_pixels

# Function to extract face from frame
def extract_face(frame, required_size=(160, 160)):
    results = detector.detect_faces(frame)
    if results:
        x1, y1, width, height = results[0]['box']
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        face = frame[y1:y2, x1:x2]
        if face.size == 0:
            return None
        face_image = Image.fromarray(face)
        face_image = face_image.resize(required_size)
        face_array = np.asarray(face_image)
        return face_array
    return None

# Function to get embeddings
def get_embedding(model, face_pixels):
    face_pixels = preprocess_face(face_pixels)
    samples = np.expand_dims(face_pixels, axis=0)
    yhat = model.predict(samples)
    return yhat[0]

# Test the SVM with training embeddings (optional)
if target_embeddings is not None:
    print("Testing SVM with known embeddings:")
    for idx, embedding in enumerate(target_embeddings):
        prediction = svm_model.predict([embedding])
        probability = svm_model.predict_proba([embedding])[0][1]
        print(f"Embedding {idx}: Prediction={prediction}, Probability={probability:.2f}")

# Initialize video capture
video_capture = cv2.VideoCapture("http://192.168.186.19:8080/video")  # Replace with your video stream URL

if not video_capture.isOpened():
    print("Error: Unable to open video stream.")
    exit()

print("Starting video stream... Press 'q' to exit.")

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to retrieve frame. Exiting...")
        break

    # Extract face from frame
    face_pixels = extract_face(frame)
    if face_pixels is not None:
        # Debug: Save and display cropped face for verification
        cv2.imwrite(f"debug_face_{np.random.randint(1000)}.jpg", face_pixels)

        # Get embedding
        embedding = get_embedding(facenet_model, face_pixels)
        embedding = embedding.reshape(1, -1)

        # Perform prediction
        prediction = svm_model.predict(embedding)
        probability = svm_model.predict_proba(embedding)[0][1]

        # Debug: Print prediction and probability
        print(f"Prediction: {prediction}, Probability: {probability:.2f}")

        # Define label and color based on prediction and probability
        if prediction == 1 and probability > 0.3:  # Adjust threshold here
            label = "Mohamad"
            color = (0, 255, 0)  # Green
        else:
            label = "Unknown"
            color = (0, 0, 255)  # Red

        # Display label on frame
        cv2.putText(frame, f"{label} ({probability:.2f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    else:
        print("No face detected.")

    # Display the video feed
    cv2.imshow('Mobile Camera Feed', frame)

    # Break on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
video_capture.release()
cv2.destroyAllWindows()
