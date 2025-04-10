import flwr as fl
import tensorflow as tf
from tensorflow import keras
import sys
import os
import numpy as np
from tensorflow.keras.utils import Sequence

# Custom data generator for YOLO annotations
class YOLODataGenerator(Sequence):
    def __init__(self, image_dir, annotation_dir, classes, input_size=(224, 224), batch_size=32):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.classes = classes
        self.input_size = input_size
        self.batch_size = batch_size
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(".jpg") or f.endswith(".png")]
        self.num_classes = len(classes)

    def __len__(self):
        return int(np.ceil(len(self.image_files) / self.batch_size))

    def __getitem__(self, index):
        batch_files = self.image_files[index * self.batch_size:(index + 1) * self.batch_size]
        images = []
        labels = []

        for file in batch_files:
            image_path = os.path.join(self.image_dir, file)
            annotation_path = os.path.join(self.annotation_dir, file.replace(".jpg", ".txt").replace(".png", ".txt"))

            # Load and preprocess the image
            image = tf.keras.utils.load_img(image_path, target_size=self.input_size)
            image = tf.keras.utils.img_to_array(image) / 255.0
            images.append(image)

            # Load and preprocess the annotation
            label = np.zeros(self.num_classes)
            with open(annotation_path, "r") as f:
                for line in f.readlines():
                    class_id, _, _, _, _ = map(float, line.strip().split())
                    label[int(class_id)] = 1  # One-hot encoding for the class
            labels.append(label)

        return np.array(images), np.array(labels)

# Define the Flower Client
class TrafficSignClient(fl.client.NumPyClient):
    def __init__(self, model, train_data, test_data):
        # Initialize the client with model, train data, and test data
        self.model = model
        self.train_data = train_data
        self.test_data = test_data

    def get_parameters(self, config=None):
        return self.model.get_weights()

    def fit(self, parameters, config):
        # Set the model's weights to the ones from the server
        print("Client received weights. Starting training...")
        self.model.set_weights(parameters)
        self.model.fit(
            self.train_data,
            epochs=1,
            validation_data=self.test_data,
            verbose=2,
        )
        print("Client finished training. Sending updated weights...")
        return self.model.get_weights(), len(self.train_data), {}

    def evaluate(self, parameters, config):
        # Set the model's weights to the ones from the server and evaluate it
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.test_data, verbose=0)
        print(f"Evaluation - Loss: {loss}, Accuracy: {accuracy}")
        return loss, len(self.test_data), {"accuracy": accuracy}

# Main Script
if __name__ == "__main__":
    # Check for client_id argument
    if len(sys.argv) < 2:
        print("Usage: python client.py <client_id>")
        sys.exit(1)

    client_id = sys.argv[1]

    # Define dataset paths
    image_dir = rf"C:\Users\DELL\Desktop\federated\itsd2\client_1\images"
    annotation_dir = rf"C:\Users\DELL\Desktop\federated\itsd2\client_1\label"
    classes = ['ALL_MOTOR_VEHICLE_PROHIBITED', 'AXLE_LOAD_LIMIT', 'BARRIER_AHEAD', 'BULLOCK_AND_HANDCART_PROHIBITED', 'BULLOCK_PROHIBITED']  # Replace with actual class names

    # Load train and test data
    train_data = YOLODataGenerator(image_dir, annotation_dir, classes, input_size=(224, 224), batch_size=32)
    test_data = YOLODataGenerator(image_dir, annotation_dir, classes, input_size=(224, 224), batch_size=32)

    # Define the model
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(len(classes), activation="softmax")
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    # Start the Flower client
    fl.client.start_numpy_client(
        server_address="localhost:5002",  # Make sure this matches the server's address
        client=TrafficSignClient(model, train_data, test_data),
    )
