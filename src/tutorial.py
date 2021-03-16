import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from PIL import Image

with open("/mnt/c/Users/Antti/Documents/Aalto/machine_learning/ocr_project/src/letter_data.csv",'r') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter = ',')
    arrays = []
    letters = []
    for line in csv_reader:
        if line[0].isnumeric():
            letters.append(line[1])
            letter_data = []
            first = 6
            last = 14
            for i in range(16):
                row = []
                help_list = line[first:last]
                
                for number in help_list:
                    if number == '1':
                        row.append(0)
                    elif number == '0':
                        row.append(255)
                
                letter_data.append(row)
                first += 8; last += 8
            letter_array = np.array(letter_data)
            arrays.append(letter_array.astype(np.float32))
        else:
            continue

data ={'letter' : letters, 'array': arrays}
df = pd.DataFrame(data)

letter = df.pop('letter')
dataset = tf.data.Dataset.from_tensor_slices((df.values, letter.values))
for feat, targ in dataset.take(5):
  print ('Features: {}, Target: {}'.format(feat, targ))