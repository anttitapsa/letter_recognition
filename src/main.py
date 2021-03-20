import csv
#import matplotlib.pyplot as plt
import numpy as np
#import pandas as pd
import tensorflow as tf

from PIL import Image

print("Libraries imported")

with open("/mnt/c/Users/Antti/Documents/Aalto/machine_learning/ocr_project/src/Own_dataset/letter_data.csv",'r') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter = ',')
    arrays = []
    letters = []
    for line in csv_reader:
        if line[0].isnumeric():
            letters.append(float(ord(line[1]))) #changed characters to ascii ints and then to floats
            letter_data = []
            first = 6
            last = 14
            for i in range(16):
                row = line[first:last]
                '''
                help_list = line[first:last]
                
                for number in help_list:
                    if number == '1':
                        row.append(0)
                    elif number == '0':
                        row.append(255)
                '''
                letter_data.append(row)
                first += 8; last += 8
            letter_array = np.array(letter_data)
            arrays.append(letter_array.astype(np.float32))
        else:
            continue

features = np.array(arrays)
labels = np.array(letters)

print("Data is prepared")

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(16, 8)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(127)
])

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

print("model is ready")
model.fit(features, labels, epochs=1)