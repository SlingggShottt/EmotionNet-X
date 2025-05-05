import cv2
import numpy as np
import dlib
import os
import math
import json
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout, BatchNormalization, add, Activation
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split

NUM_CLASSES = 7  # Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral

# Function to load configuration from a JSON file
def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

# Load FER-2013 dataset with error handling
def load_fer2013(data_path):
    data = []
    labels = []
    
    # Map emotion labels to integers
    emotion_to_int = {
        'angry': 0,
        'disgust': 1,
        'fear': 2,
        'happy': 3,
        'sad': 4,
        'surprise': 5,
        'neutral': 6
    }
    
    for emotion in os.listdir(data_path):
        emotion_path = os.path.join(data_path, emotion)
        if not os.path.isdir(emotion_path):
            continue
        for image_name in os.listdir(emotion_path):
            image_path = os.path.join(emotion_path, image_name)
            try:
                image = cv2.imread(image_path, cv2.IMREAD_COLOR)
                if image is None:
                    print(f"Warning: {image_path} could not be read.")
                    continue
                if len(image.shape) == 2 or image.shape[-1] == 1:
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                # Do not resize the image: use its original dimensions
                data.append(image)
                labels.append(emotion_to_int[emotion])
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
    return data, np.array(labels)

# Preprocess images
def preprocess_image(image):
    # Ensure the image is in RGB
    if len(image.shape) == 2 or image.shape[-1] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # Convert to YCrCb for luminance equalization
    img_ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    channels = cv2.split(img_ycrcb)
    channels[0] = cv2.equalizeHist(channels[0])
    img_ycrcb_eq = cv2.merge(channels)
    # Convert back to RGB
    image_eq = cv2.cvtColor(img_ycrcb_eq, cv2.COLOR_YCrCb2RGB)
    
    # Normalize to [0, 1]
    image_eq = image_eq.astype('float32') / 255.0
    return image_eq

# Pad all images to the maximum height and width found in the dataset
def pad_images_to_max(images):
    max_h = max(image.shape[0] for image in images)
    max_w = max(image.shape[1] for image in images)
    padded_images = []
    for img in images:
        h, w, _ = img.shape
        pad_h = max_h - h
        pad_w = max_w - w
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        padded = cv2.copyMakeBorder(img, top, bottom, left, right,
                                    cv2.BORDER_CONSTANT, value=[0, 0, 0])
        padded_images.append(padded)
    return np.array(padded_images), (max_h, max_w)

# Define a helper function for a residual block
def residual_block(x, filters, kernel_size=3):
    shortcut = x
    # First convolution layer
    x = Conv2D(filters, kernel_size, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    # Second convolution layer without activation
    x = Conv2D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    # Add skip connection and apply activation
    x = add([shortcut, x])
    x = Activation('relu')(x)
    return x

# Build the CNN model with residual connections using Global Average Pooling
def build_residual_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    
    # Block 1
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = residual_block(x, 32)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)
    
    # Block 2
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = residual_block(x, 64)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)
    
    # Block 3
    x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = residual_block(x, 128)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)
    
    # Classification head using Global Average Pooling to allow for variable image sizes
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Main function
def main():
    # Load configuration from 'config.json'
    config = load_config("config.json")
    batch_size = config.get("batch_size", 64)
    epochs = config.get("epochs", 50)
    data_path = config.get("data_path", "datasets/FER-2013/train")
    
    # Load and preprocess data
    data, labels = load_fer2013(data_path)
    data = [preprocess_image(image) for image in data]
    # Pad images so they all have the same dimensions
    data, padded_size = pad_images_to_max(data)
    
    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, random_state=42)
    
    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1,
        brightness_range=[0.8, 1.2]
    )
    datagen.fit(X_train)
    
    # Build the model using the padded image dimensions
    model = build_residual_model((padded_size[0], padded_size[1], 3), NUM_CLASSES)
    
    # Calculate steps per epoch to include all samples
    steps_per_epoch = math.ceil(len(X_train) / batch_size)
    
    # Set up callbacks
    checkpoint = ModelCheckpoint('best_emotion_model.h5', monitor='val_accuracy', save_best_only=True, verbose=1)
    early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
    
    # Train the model using the data generator
    model.fit(datagen.flow(X_train, y_train, batch_size=batch_size),
              validation_data=(X_val, y_val),
              epochs=epochs,
              steps_per_epoch=steps_per_epoch,
              callbacks=[checkpoint, early_stop])
    
    # Save the final model
    model.save("emotion.h5")

if __name__ == "__main__":
    main()
