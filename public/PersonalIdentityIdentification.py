import os
import keras
import tensorflow as tf
from keras import layers
from keras._tf_keras.keras.preprocessing.image import load_img, img_to_array
from file_handler import extract_text
import numpy as np
<<<<<<< HEAD:public/PersonalIdentityIdentification.py
import pandas as pd
from detector import detect_ids
=======
import fitz  # PyMuPDF
from pdf2image import convert_from_path
>>>>>>> 20b8a875794846b9580b91d27d77c4aaaf09bf48:main.py

positive_samples = [
    "object_detection/imgs/img1.png",
    "object_detection/imgs/img2.png",
    "object_detection/imgs/img3.png",
    "object_detection/imgs/img4.png",
    "object_detection/imgs/img5.png",
    "object_detection/imgs/img6.jpg",
    "object_detection/imgs/img7.jpg",
    "object_detection/imgs/img8.jpg",
    "object_detection/imgs/img9.jpg"
]
negative_samples = [
    'object_detection/negative_imgs/img1.jpg',
    "object_detection/negative_imgs/img2.jpg",
    "object_detection/negative_imgs/img3.jpg",
    "object_detection/negative_imgs/img4.jpg",
    "object_detection/negative_imgs/img5.jpg",
    "object_detection/negative_imgs/img6.jpg",
    "object_detection/negative_imgs/img7.jpg",
    "object_detection/negative_imgs/img8.jpg",
    "object_detection/negative_imgs/img9.jpg"
]



# Function to load images
def load_images(image_paths, label):
    images = []
    labels = []
    for img_path in image_paths:
        img = load_img(img_path, target_size=(224, 224))
        img_array = img_to_array(img) / 255.0  # Normalize the image to [0, 1]
        images.append(img_array)
        labels.append(label)
    return np.array(images), np.array(labels)


def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0  # Normalize the image to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions to match the input shape (1, 224, 224, 3)
    return img_array


# Function to convert PDF pages to images
def convert_pdf_to_images(pdf_path):
    images = convert_from_path(pdf_path)
    image_paths = []
    for i, image in enumerate(images):
        image_path = f'page_{i + 1}.png'
        image.save(image_path, 'PNG')
        image_paths.append(image_path)
    return image_paths


# Function to extract embedded images from PDF
def extract_images_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    extracted_image_paths = []

    # Create a directory to save extracted images
    if not os.path.exists('extracted_images'):
        os.makedirs('extracted_images')

    # Loop through each page in the PDF
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)  # Load the page
        images = page.get_images(full=True)  # Get all images on the page

        # Loop through the images on the page
        for img_index, img in enumerate(images):
            xref = img[0]  # XREF of the image
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]  # Extract the image bytes
            image_ext = base_image["ext"]  # Extract the image format/extension
            image_path = f"extracted_images/page_{page_num + 1}_image_{img_index + 1}.{image_ext}"

            # Save the extracted image
            with open(image_path, "wb") as img_file:
                img_file.write(image_bytes)
            extracted_image_paths.append(image_path)

    return extracted_image_paths


# Function to load images for model input
def load_and_preprocess_images(image_paths):
    images = []
    for img_path in image_paths:
        img = load_img(img_path, target_size=(224, 224))
        img_array = img_to_array(img) / 255.0  # Normalize the image to [0, 1]
        images.append(img_array)
    return np.array(images)


# Function to process and classify images from a PDF
def process_pdf_images_for_model(pdf_path, model):
    # Convert PDF pages to images
    page_images = convert_pdf_to_images(pdf_path)
    # Extract embedded images from PDF
    embedded_images = extract_images_from_pdf(pdf_path)

    # Load and preprocess all images
    all_images = load_and_preprocess_images(page_images + embedded_images)

    # Predict using the trained model
    predictions = model.predict(all_images)

    # Interpret predictions
    for i, prediction in enumerate(predictions):
        predicted_class = np.argmax(prediction)  # Get the index of the highest probability
        if predicted_class == 1:
            print(f"Image {i + 1} is predicted to be a positive sample.")
        else:
            print(f"Image {i + 1} is predicted to be a negative sample.")


# Load positive and negative images
positive_images, positive_labels = load_images(positive_samples, 1)
negative_images, negative_labels = load_images(negative_samples, 0)

# Combine the datasets
images = np.concatenate((positive_images, negative_images), axis=0)
labels = np.concatenate((positive_labels, negative_labels), axis=0)

# Create a TensorFlow dataset
dataset = tf.data.Dataset.from_tensor_slices((images, labels))
dataset = dataset.shuffle(buffer_size=len(images)).batch(32)

# Create a TensorFlow model
model = keras.Sequential([
    layers.Conv2D(32, 3, activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(2, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(dataset, epochs=10)

<<<<<<< HEAD:public/PersonalIdentityIdentification.py
input_image_path = "C:/object_detection/WhatsApp Image 2024-08-21 at 15.57.49_cd148554.jpg" # Replace with your input image path

# Make a prediction
input_image = preprocess_image(input_image_path)
prediction = model.predict(input_image)

positive_percentage = prediction[0][1] * 100  # Percentage for the positive class
negative_percentage = prediction[0][0] * 100  # Percentage for the negative class



    # Interpret the prediction
predicted_class = np.argmax(prediction)  # Get the index of the highest probability

if predicted_class == 1:
    text = extract_text(input_image_path)
    detect_ids(text)
else:
    print("No Personal Identity Information given")
=======
# Example: Process a PDF file and classify its contents
pdf_path = "Adhr Crd_Purnima.pdf"  # Replace with the path to your PDF file
process_pdf_images_for_model(pdf_path, model)
>>>>>>> 20b8a875794846b9580b91d27d77c4aaaf09bf48:main.py
