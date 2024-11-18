import os

import cv2
from albumentations import (
	Compose, Rotate, RandomScale, HorizontalFlip, VerticalFlip
)
from albumentations import Resize
from tqdm import tqdm


def augment_images(input_dir, output_dir, num_augmented_images=5):
	# Ensure output directory exists
	os.makedirs(output_dir, exist_ok=True)

	# Define augmentation pipeline
	transform = Compose([
		Rotate(limit=30, p=0.5),
		RandomScale(scale_limit=0.2, p=0.5),
		HorizontalFlip(p=0.5),
		VerticalFlip(p=0.5),
		Resize(height=256, width=256)  # Resize to desired dimensions
	])

	# Process each image in the input directory
	for filename in tqdm(os.listdir(input_dir), desc="Processing images"):
		if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
			image_path = os.path.join(input_dir, filename)
			image = cv2.imread(image_path)

			# Generate augmented images
			for i in range(num_augmented_images):
				augmented = transform(image=image)
				augmented_image = augmented['image']
				augmented_filename = (f"{os.path.splitext(filename)[0]}_aug_"
									  f"{i}.jpg")
				augmented_path = os.path.join(output_dir, augmented_filename)
				cv2.imwrite(augmented_path, augmented_image)


if __name__ == "__main__":
	input_directory = './datasets/mohamad'
	output_directory = './datasets/mohamad_augmentation'
	augment_images(input_directory, output_directory)
