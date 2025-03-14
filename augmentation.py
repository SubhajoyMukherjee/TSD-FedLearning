from PIL import Image
import numpy as np
import os
import random
import cv2
from PIL import ImageOps, ImageEnhance

# Augmentation functions
def flip_image(image):
    return ImageOps.mirror(image)

def gamma_correction(image, gamma=1.5):
    inv_gamma = 1.0 / gamma
    table = [((i / 255.0) ** inv_gamma) * 255 for i in range(256)]
    table = np.array(table, np.uint8)
    return Image.fromarray(cv2.LUT(np.array(image), table))

def add_noise(image):
    np_image = np.array(image)
    noise = np.random.normal(0, 25, np_image.shape).astype(np.uint8)
    noisy_image = cv2.add(np_image, noise)
    return Image.fromarray(noisy_image)

def rotate_image(image):
    angle = random.choice([90, 180, 270])
    return image.rotate(angle)

def scale_image(image, scale_range=(0.8, 1.2)):
    scale = random.uniform(*scale_range)
    width, height = image.size
    new_size = (int(width * scale), int(height * scale))
    resized_image = image.resize(new_size, Image.Resampling.LANCZOS)
    left = max((new_size[0] - width) // 2, 0)
    top = max((new_size[1] - height) // 2, 0)
    return resized_image.crop((left, top, left + width, top + height))

def zoom_image(image):
    zoom_factor = random.uniform(1.0, 1.5)
    width, height = image.size
    crop_width = int(width / zoom_factor)
    crop_height = int(height / zoom_factor)
    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    cropped_image = image.crop((left, top, left + crop_width, top + crop_height))
    return cropped_image.resize((width, height), Image.Resampling.LANCZOS)

def adjust_brightness(image):
    enhancer = ImageEnhance.Brightness(image)
    factor = random.uniform(0.5, 1.5)
    return enhancer.enhance(factor)

def distort_image(image):
    width, height = image.size
    src = np.float32([[0, 0], [width - 1, 0], [0, height - 1], [width - 1, height - 1]])
    dst = src + np.random.uniform(-20, 20, src.shape).astype(np.float32)
    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(np.array(image), matrix, (width, height))
    return Image.fromarray(warped)

# Combine augmentations
def augment_image(image, backgrounds_dir=None):
    operations = [
        flip_image,
        gamma_correction,
        add_noise,
        rotate_image,
        scale_image,
        zoom_image,
        adjust_brightness,
        distort_image,
    ]

    random.shuffle(operations)
    for operation in operations[:random.randint(1, len(operations))]:
        image = operation(image)
    return image

# Dataset augmentation function
def augment_dataset_with_percentage(train_dir, augmented_dir, percentage=0.3, backgrounds_dir=None):
    os.makedirs(augmented_dir, exist_ok=True)
    print(f"Augmented folder created at: {augmented_dir}")

    for folder_name in os.listdir(train_dir):
        source_folder = os.path.join(train_dir, folder_name)
        target_folder = os.path.join(augmented_dir, folder_name)
        os.makedirs(target_folder, exist_ok=True)
        print(f"Processing folder: {source_folder}")

        images = [img for img in os.listdir(source_folder) if img.lower().endswith(('.jpg', '.jpeg', '.png'))]
        folder_total = len(images)
        if folder_total == 0:
            print(f"No images found in {source_folder}. Skipping...")
            continue

        # Calculate number of images to generate
        num_to_augment = int(folder_total * percentage)
        print(f"Augmenting {num_to_augment} images for folder: {folder_name}")

        for i in range(num_to_augment):
            image_name = random.choice(images)
            image_path = os.path.join(source_folder, image_name)
            try:
                image = Image.open(image_path)
                augmented_image = augment_image(image, backgrounds_dir)

                save_path = os.path.join(target_folder, f"aug_{i + 1}_{image_name}")
                augmented_image.save(save_path)
                print(f"Saved augmented image to: {save_path}")

            except Exception as e:
                print(f"Error processing image {image_name}: {e}")

# Main function
if __name__ == "__main__":
    # Directory paths
    train_directory = r'D:\FEDERATED LEARNING\PlantVillage\train'
    augmented_directory = r'D:\FEDERATED LEARNING\PlantVillage\augmented'
    backgrounds_directory = r'D:\FEDERATED LEARNING\backgrounds'

    # Augment dataset by 30%
    augment_dataset_with_percentage(train_directory, augmented_directory, percentage=0.3, backgrounds_dir=backgrounds_directory)
