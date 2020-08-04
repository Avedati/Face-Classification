import tensorflow as tf
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPool2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

img_width = 256 # TODO: change this
img_height = 256 # TODO: change this

datagen = ImageDataGenerator(rescale=1/255.0, validation_split=0.2)
training_data_generator = datagen.flow_from_directory(directory='real_and_fake_face', target_size=(img_width, img_height), class_mode='binary', batch_size=16, subset='training')
validation_data_generator = datagen.flow_from_directory(directory='real_and_fake_face', target_size=(img_width, img_height), class_mode='binary', batch_size=16, subset='validation')

model = Sequential([
	Conv2D(16, (3, 3), input_shape=(img_width, img_height, 3), activation='relu'),
	MaxPool2D(2, 2),
	Dropout(0.2),

	Conv2D(32, (3, 3), activation='relu'),
	MaxPool2D(2, 2),
	Dropout(0.3),

	Conv2D(64, (3, 3), activation='relu'),
	MaxPool2D(2, 2),
	Dropout(0.5),

	Flatten(),
	Dense(128, activation='relu'),
	Dense(128, activation='relu'),
	Dense(128, activation='relu'),
	Dropout(0.5),
	Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(training_data_generator, steps_per_epoch=len(training_data_generator), epochs=15, validation_data=validation_data_generator, validation_steps=len(validation_data_generator))
