import pickle

import cv2
import numpy as np
from PIL import Image
from keras.models import load_model
from mtcnn import MTCNN


model = load_model('facenet_keras.h5')  # Set compile=False if weights are separate
# model.load_weights('weights.h5')

# Load the trained SVM classifier
with open('mohamad.pkl', 'rb') as f:
	svm_model = pickle.load(f)

# Initialize MTCNN face detector
detector = MTCNN()


def extract_face(frame, required_size=(160, 160)):
	# Detect faces in the frame
	results = detector.detect_faces(frame)
	if results:
		x1, y1, width, height = results[0]['box']
		x1, y1 = abs(x1), abs(y1)
		x2, y2 = x1 + width, y1 + height
		face = frame[y1:y2, x1:x2]
		# Resize face to required size
		face_image = Image.fromarray(face)
		face_image = face_image.resize(required_size)
		face_array = np.asarray(face_image)
		return face_array
	return None


def get_embedding(model, face_pixels):
	# Standardize pixel values
	face_pixels = face_pixels.astype('float32')
	mean, std = face_pixels.mean(), face_pixels.std()
	face_pixels = (face_pixels - mean) / std
	samples = np.expand_dims(face_pixels, axis=0)
	# Get embedding
	yhat = model.predict(samples)
	return yhat[0]


# Initialize webcam
# video_capture = cv2.VideoCapture(0)


# URL of the mobile camera stream
video_stream_url = "http://192.168.186.19:8080/video"

# Open the video stream
video_capture = cv2.VideoCapture(video_stream_url)

if not video_capture.isOpened():
    print("Error: Unable to open video stream.")
    exit()


while True:
	ret, frame = video_capture.read()
	if not ret:
		break

	face_pixels = extract_face(frame)
	if face_pixels is not None:
		embedding = get_embedding(model, face_pixels)
		embedding = np.expand_dims(embedding, axis=0)
		prediction = svm_model.predict(embedding)
		probability = svm_model.predict_proba(embedding)[0][1]

		if prediction == 1 and probability > 0.5:
			label = "Mohamad"
			color = (0, 255, 0)
		else:
			label = "Unknown"
			color = (0, 0, 255)

		cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
					color,
					2)

	cv2.imshow('Real-time Face Recognition', frame)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

video_capture.release()
cv2.destroyAllWindows()
