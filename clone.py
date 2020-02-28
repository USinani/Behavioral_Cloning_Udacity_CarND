import csv
import cv2
import numpy as np

lines = []

with open('./data/driving_log.csv') as csvf:
    reader = csv.reader(csvf)
    for line in reader:
        lines.append(line)
        
images = []
measurements = []

for line in lines:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = './data/IMG'+ filename
    image = cv2.imread(current_path)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)
    

X_train = np.array(images)
Y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers import flatten, Dense

model = Sequential()
model.add(flatten(input_shape(160, 320, 3)))
model.add(Dense(1))

model.compile(loss='mse', optimizers = 'adam')
model.fit(X_train, Y_train, validation_split=0.2, shuffle=True, nb_epoch = 7)
model.save('model.h5')
