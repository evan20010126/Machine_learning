import os
import pathlib

import matplotlib
import matplotlib.pyplot as plt

import io
import scipy.misc
import numpy as np
from six import BytesIO
from PIL import Image, ImageDraw, ImageFont
from six.moves.urllib.request import urlopen

import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd

tf.get_logger().setLevel('ERROR')

# Input video的資料夾路徑
dirPath = r'C:\Users\User\Downloads\archive (2)\data\training_images'

csv_path = r'C:\Users\User\Downloads\archive (2)\data'

# 列出指定路徑底下所有檔案(包含資料夾)
allFileList = os.listdir(dirPath)

df = pd.read_csv(f'{csv_path}\\train_solution_bounding_boxes.csv')

print(df.iloc[0])

train_x = np.array([])
train_y = np.array([])

count = 0
print(allFileList)
for f in allFileList:
    # print(Image.open(f"{dirPath}\\{f}"))
    # print(np.array(Image.open(f"{dirPath}\\{f}")).shape)
    train_x = np.append(
        train_x, [np.array(Image.open(f"{dirPath}\\{f}"))])
    temp = np.array([df.iloc[count]["xmin"], df.iloc[count]["ymin"],
                    df.iloc[count]["xmax"], df.iloc[count]["ymax"]])
    train_y = np.append(train_y, temp)
    count += 1


# train_x = np.array([np.array(Image.open("vid_4_600.jpg")),
#                    np.array(Image.open("vid_4_600.jpg"))])

train_x = np.reshape(train_x, (-1, 380, 676, 3))
print(train_x.shape)
train_y = np.reshape(train_y, (-1, 4))
print(train_y.shape)

# train_y = np.array([np.array([286.6397, 187.5241, 407.9479, 232.0286]), np.array(
#     [286.6397, 187.5241, 407.9479, 232.0286])])
# print(train_y.shape)

x_test = 0
y_test = 0

# conv1 =
# conv1 = tf.keras.layers.BatchNormalization()(conv1)
# tf.keras.layers.Conv2D(256, (1, 1), padding='same', activation='relu')
my_model = tf.keras.models.Sequential()
my_model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=(
    3, 3), padding="same", activation='relu', input_shape=train_x.shape[1:]))  # [380, 676, 3]
my_model.add(tf.keras.layers.BatchNormalization())
my_model.add(tf.keras.layers.Conv2D(
    16, (5, 5), dilation_rate=6, padding='same', activation='relu'))
# my_model.add(tf.keras.layers.Conv2D(
#     64, (3, 3), dilation_rate=18, padding='same', activation='relu'))
# my_model.add(tf.keras.layers.Conv2D(
#     64, (3, 3), padding='same', activation='relu'))
# my_model.add(tf.keras.layers.GlobalAveragePooling1D())
my_model.add(tf.keras.layers.Flatten())
my_model.add(tf.keras.layers.Dense(4, activation='relu'))
# my_model.add(tf.keras.layers.MaxPooling2D((2, 2), (2, 2), padding='same'))


my_model.compile(loss='categorical_crossentropy',
                 optimizer='adam', metrics=['accuracy'])
history = my_model.fit(train_x, train_y, batch_size=10, epochs=10)

score = my_model.evaluate(x_test, y_test)
print("Total loss on testing set:", score[0])
print("Accuracy of testing set:", score[1])

# print(history)
print("-"*100)
my = my_model.predict(train_x)
print(my)
