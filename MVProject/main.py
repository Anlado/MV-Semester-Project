#ResNet50 imports
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#Helpers
import os
import h5py
import math
import numpy as np
import pandas as pd
import cv2
import shutil
from PIL import Image


gpus = tf.config.list_physical_devices('GPU')
print("Num GPUs Available:", len(gpus))
if gpus:
    print("GPU Name:", gpus[0])

print("TensorFlow version:", tf.__version__)

########################################
#Remove noisy and bad images
########################################
def moveBadFiles(imgPath, destination="C:/Users/mithi/PycharmProjects/MVProject/FlaggedImages/"):
    os.makedirs(destination, exist_ok=True)
    try:
        path = os.path.join(destination, os.path.basename(imgPath))

        if os.path.exists(path):
            base, ext = os.path.splitext(path)
            i = 1
            while os.path.exists(f"{base}_{i}{ext}"):
                i += 1
            path = f"{base}_{i}{ext}"

        if os.path.exists(imgPath):
            shutil.move(imgPath, path)
        else:
            print(f"File {imgPath} not found.")

    except Exception as e:
        print(f"Error moving {imgPath}: {e}")

def isBlurry(imgPath, threshold = 0):
  image = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)
  if image is None:
    return True
  variance = cv2.Laplacian(image, cv2.CV_64F).var()
  return variance < threshold

def isNoisy(imgPath, meanThresh = 0, varThresh = 0):
  image = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)
  if image is None:
    return True
  mean, stdDev = cv2.meanStdDev(image)
  variance = stdDev[0][0] ** 2
  return variance > varThresh or mean > meanThresh

def removeBadFiles(path):
  noisy = set()
  for root, _, files in os.walk(path):
    for file in files:
      if (file.lower().endswith(("jpg", "jpeg", "png"))):
        filePath = os.path.join(root, file)
        try:
          img = Image.open(filePath)
          img.verify()
        except (IOError, SyntaxError) as e:
          noisy.add(filePath)
          continue
        if (isBlurry(filePath) or isNoisy(filePath)):
          noisy.add(filePath)

  for f in noisy:
    moveBadFiles(f)

datasetPath = "C:/Users/mithi/Downloads/Dataset"
#removeBadFiles(datasetPath)

##################################################
#Sort Images
#################################################3


removeBadFiles("/content/my_data")




############################################
#creating training and validation set
#############################################
dataDir = "C:/Users/mithi/Downloads/Dataset"


trainingDS = image_dataset_from_directory(
    dataDir,
    validation_split = .2,
    subset = "training",
    seed = 123,
    image_size = (224,224),
    batch_size = 16
)


validationDS = image_dataset_from_directory(
    dataDir,
    validation_split = .2,
    subset = "validation",
    seed = 123,
    image_size = (224,224),
    batch_size = 16
)


classNames = trainingDS.class_names
Autotune = tf.data.AUTOTUNE
trainingDS = trainingDS.cache().prefetch(buffer_size=Autotune)
validationDS = validationDS.cache().prefetch(buffer_size=Autotune)

with open("classNames.json", "w") as f:
  json.dump(classNames, f)


'''
############################################
#Training model
#############################################
#Creating a model trained off of Stanford Dog dataset
baseModel = ResNet50(weights = "imagenet", include_top = False, input_shape = (224,224,3))
baseModel.trainable = False

model = models.Sequential([
    baseModel,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation = "relu"),
    layers.Dropout(.5),
    layers.Dense(len(classNames), activation = "softmax")
])

model.compile(
    optimizer = "adam",
    loss = "sparse_categorical_crossentropy",
    metrics = ["accuracy"]
)

history = model.fit(
    trainingDS,
    validation_data = validationDS,
    epochs=15
)

model.save("C:/Users/mithi/PycharmProjects/MVProject/StanfordDogBreedModel.keras")

'''
modelPath = "C:/Users/mithi/Downloads/4Epoch/StanfordDogBreedModel.keras"
model = tf.keras.models.load_model(modelPath)


def prepareImage(imgPath):
    img = image.load_img(imgPath, target_size=(224, 224))
    imgArray = image.img_to_array(img)
    imgArray = np.expand_dims(imgArray, axis=0)
    imgArray /= 255.0
    return imgArray

# Upload and load model
images = [
"C:/Users/mithi/PycharmProjects/MVProject/Chart_rosyjski_borzoj_rybnik-kamien_pl.jpg",
"C:/Users/mithi/PycharmProjects/MVProject/Sphynx_92.jpg"
]

for imgPath in images:
    img = prepareImage(imgPath)
    prediction = model.predict(img)
    predictedIndex = np.argmax(prediction)
    confidence = np.max(prediction)

    print(f"Predicted breed: {classNames[predictedIndex]} ({confidence * 100:.2f}% confidence)")

    if confidence > 0.5:
        print(f"Predicted breed: {classNames[predictedIndex]} ({confidence * 100:.2f}% confidence)")
    else:
        print("Unknown cat or dog")



