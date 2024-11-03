import os
import cv2
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import normalize, to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
import matplotlib.pyplot as plt

# Function to load and preprocess images
def load_images(image_directory):
    dataset = []
    label = []

    INPUT_SIZE = 64
    ind = 1

    yes_tumor_images = os.listdir(image_directory + 'yes/')
    no_tumor_images = os.listdir(image_directory + 'no/')

    for types in yes_tumor_images:
        for i, image_name in enumerate(os.listdir(image_directory + 'yes/' + types)):
            if(image_name.split('.')[1] == 'jpg'):
                image = cv2.imread(image_directory + 'yes/' + types + '/' + image_name)
                image = Image.fromarray(image, 'RGB')
                image = image.resize((INPUT_SIZE, INPUT_SIZE))
                dataset.append(np.array(image))
                label.append(ind)
        ind += 1

    for i, image_name in enumerate(no_tumor_images):
        if(image_name.split('.')[1] == 'jpg'):
            image = cv2.imread(image_directory + 'no/' + image_name)
            image = Image.fromarray(image, 'RGB')
            image = image.resize((INPUT_SIZE, INPUT_SIZE))
            dataset.append(np.array(image))
            label.append(0)

    dataset = np.array(dataset)
    label = np.array(label)

    return dataset, label

# Load and preprocess images
dataset, label = load_images('Training/')

# Split dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(dataset, label, test_size=0.2, random_state=0)

# Normalize pixel values
x_train = normalize(x_train, axis=1)
x_test = normalize(x_test, axis=1)

# One-hot encode labels
y_train = to_categorical(y_train, num_classes=4)
y_test = to_categorical(y_test, num_classes=4)

# Model architecture
model = Sequential()
model.add(Conv2D(32, (3,3), input_shape=(64, 64, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32, (3,3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(4))
model.add(Activation('softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit(x_train, y_train, batch_size=16, verbose=1, epochs=10, validation_data=(x_test, y_test), shuffle=False)
# Evaluate the model on the test set
accuracy = model.evaluate(x_test, y_test)[1]
print("Accuracy:", accuracy)
# Plot training and validation accuracy/loss curves
plt.figure(figsize=(12, 6))

# Plot training & validation accuracy values
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')

# Save the plot as a JPG image in the static directory
plt.savefig('static/accuracy_loss_curves.jpg')
