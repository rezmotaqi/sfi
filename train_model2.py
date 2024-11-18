import os
import pickle

import face_recognition
import numpy as np
from sklearn.svm import SVC
from tqdm import tqdm


def process_images_from_directory(directory):
	"""
	Processes images in the given directory to extract face embeddings.

	Args:
		directory (str): Path to the directory containing images.

	Returns:
		tuple: A tuple containing:
			- embeddings (np.ndarray): Array of face embeddings.
			- image_paths (list): List of image file paths corresponding to
			the embeddings.
	"""
	embeddings = []
	image_paths = []
	# Traverse the directory
	for root, _, files in os.walk(directory):
		# Filter image files
		image_files = [file for file in files if
					   file.lower().endswith(('.jpg', '.jpeg', '.png'))]
		# Process each image file with a progress bar
		for file in tqdm(image_files, desc=f"Processing images in {root}",
						 unit="image"):
			image_path = os.path.join(root, file)
			try:
				image = face_recognition.load_image_file(image_path)
				face_locations = face_recognition.face_locations(image,
																 model='cnn')
				face_encodings = face_recognition.face_encodings(image,
																 face_locations)
				if face_encodings:
					embeddings.append(face_encodings[0])
					image_paths.append(image_path)
				else:
					print(f"No face found in image: {image_path}")
			except Exception as e:
				print(f"Error processing image {image_path}: {e}")
	return np.array(embeddings), image_paths


# Paths to the positive and negative image directories
positive_image_folder = './datasets/mohamad'
negative_image_folder = './datasets/negative_small'

# Process positive images
print("Processing positive images...")
positive_embeddings, positive_image_paths = process_images_from_directory(
	positive_image_folder)

# Process negative images
print("Processing negative images...")
negative_embeddings, negative_image_paths = process_images_from_directory(
	negative_image_folder)

# Check if embeddings are non-empty
if positive_embeddings.size == 0:
	raise ValueError("No embeddings found in the positive images.")
if negative_embeddings.size == 0:
	raise ValueError("No embeddings found in the negative images.")

# Combine embeddings and create labels
X = np.concatenate((positive_embeddings, negative_embeddings))
y = np.concatenate(
	(np.ones(len(positive_embeddings)), np.zeros(len(negative_embeddings))))

# Train the classifier
print("Training the classifier...")
classifier = SVC(kernel='linear', probability=True)
classifier.fit(X, y)

# Save the trained classifier and target embeddings
model_path = 'mohamad.pkl'
with open(model_path, 'wb') as f:
	pickle.dump(
		{'classifier': classifier, 'target_embeddings': positive_embeddings},
		f)

print("Model trained and saved successfully.")
