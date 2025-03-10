import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os
from PIL import Image
from sklearn.model_selection import train_test_split


# U-Net Model Definition
def unet_model(input_size=(1024, 1024, 1)):
    inputs = layers.Input(input_size)

    # Downsampling
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
    p4 = layers.MaxPooling2D((2, 2))(c4)

    c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(p4)
    c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(c5)

    # Upsampling
    u6 = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = layers.concatenate([u6, c4])
    c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(u6)
    c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c6)

    u7 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(u7)
    c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c7)

    u8 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u8)
    c8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c8)

    u9 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = layers.concatenate([u9, c1])
    c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u9)
    c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c9)

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = models.Model(inputs=[inputs], outputs=[outputs])
    return model


# Load and preprocess data
def load_and_preprocess_image(image_path, target_size=(1024, 1024)):
    img = Image.open(image_path).convert('L')  # Convert image to grayscale
    img = img.resize(target_size)  # Resize image to target size
    img = np.array(img) / 255.0  # Normalize image to [0, 1]
    img = np.expand_dims(img, axis=-1)  # Add channel dimension
    return img


def load_data(masked_images_dir, real_images_dir, target_size=(1024, 1024)):
    masked_images = []
    real_images = []

    masked_images_paths = [os.path.join(masked_images_dir, fname) for fname in os.listdir(masked_images_dir)]
    real_images_paths = [os.path.join(real_images_dir, fname) for fname in os.listdir(real_images_dir)]

    for masked_path, real_path in zip(masked_images_paths, real_images_paths):
        masked_img = load_and_preprocess_image(masked_path, target_size)
        real_img = load_and_preprocess_image(real_path, target_size)

        masked_images.append(masked_img)
        real_images.append(real_img)

    return np.array(masked_images), np.array(real_images)


# Compile and train the model
def train_model(model, masked_images, real_images, epochs=50, batch_size=2):
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    history = model.fit(masked_images, real_images, epochs=epochs, batch_size=batch_size, validation_split=0.1)

    return model, history


# Plot training history
def plot_history(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()


if __name__ == "__main__":
    masked_images, real_images = load_data('/content/drive/MyDrive/data/masks',
                                           '/content/drive/MyDrive/data/images/train')

    # Split data into training and test sets
    masked_train, masked_test, real_train, real_test = train_test_split(masked_images, real_images, test_size=0.1,
                                                                        random_state=42)

    model = unet_model()
    model, history = train_model(model, masked_train, real_train, epochs=50, batch_size=2)

    plot_history(history)

    # Evaluate the model on the test set
    test_loss, test_accuracy = model.evaluate(masked_test, real_test)
    print(f'Test loss: {test_loss}')
    print(f'Test accuracy: {test_accuracy}')

    # Save the trained model
    model.save('unet_model.h5')
