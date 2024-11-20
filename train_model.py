import os
import pickle

import cv2
import face_recognition
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.svm import SVC


# Step 1: Process the image archive to extract face embeddings
def process_image_archive(image_folder):
	embeddings = []
	image_paths = []
	for root, _, files in os.walk(image_folder):
		for file in files:
			if file.lower().endswith(('.jpg', '.jpeg', '.png')):
				image_path = os.path.join(root, file)
				image = face_recognition.load_image_file(image_path)
				face_locations = face_recognition.face_locations(image,
																 model='cnn')
				face_encodings = face_recognition.face_encodings(image,
																 face_locations)
				for encoding in face_encodings:
					embeddings.append(encoding)
					image_paths.append(image_path)
	return np.array(embeddings), image_paths


# Step 2: Cluster the embeddings to identify unique individuals
def cluster_embeddings(embeddings, eps=0.5, min_samples=5):
	clustering_model = DBSCAN(eps=eps, min_samples=min_samples,
							  metric='euclidean')
	labels = clustering_model.fit_predict(embeddings)
	return labels


# Step 3: Select the cluster corresponding to the target individual
def select_target_cluster(labels, image_paths):
	unique_labels = set(labels)
	for label in unique_labels:
		if label == -1:
			continue  # Skip noise
		print(f"Cluster {label}:")
		cluster_image_paths = [path for i, path in enumerate(image_paths) if
							   labels[i] == label]
		for path in cluster_image_paths[
					:5]:  # Display first 5 images in the cluster
			print(f" - {path}")
		user_input = input("Is this the target individual? (y/n): ")
		if user_input.lower() == 'y':
			return label
	return None


# Step 4: Train a classifier to recognize the target individual
def train_classifier(embeddings, labels, target_label):
	target_embeddings = embeddings[labels == target_label]
	non_target_embeddings = embeddings[labels != target_label]
	X = np.concatenate((target_embeddings, non_target_embeddings))
	y = np.concatenate((np.ones(len(target_embeddings)),
						np.zeros(len(non_target_embeddings))))
	classifier = SVC(kernel='linear', probability=True)
	classifier.fit(X, y)
	return classifier


# Step 5: Save the trained classifier and target embeddings
def save_model(classifier, target_embeddings,
			   model_path='mohamad.pkl'):
	with open(model_path, 'wb') as f:
		pickle.dump(
			{'classifier': classifier, 'target_embeddings':
                target_embeddings},
			f)


# Main execution
image_folder = './datasets'
embeddings, image_paths = process_image_archive(image_folder)
labels = cluster_embeddings(embeddings)
target_label = select_target_cluster(labels, image_paths)
if target_label is not None:
	classifier = train_classifier(embeddings, labels, target_label)
	save_model(classifier, embeddings[labels == target_label])
	print("Model trained and saved successfully.")
else:
	print("Target individual not identified.")
