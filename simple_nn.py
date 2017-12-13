# https://www.pyimagesearch.com/2016/11/14/installing-keras-with-tensorflow-backend/
# https://www.pyimagesearch.com/2016/09/26/a-simple-neural-network-with-python-and-keras/

from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.optimizers import SGD
from keras.utils import np_utils
from imutils import paths
import numpy as np
import argparse
import cv2
import os, sys
import pandas as pd


def image_to_feature_vector(image, size=(32, 32)):
    return cv2.resize(image, size).flatten()

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=False, help="path to input dataset")
ap.add_argument("-c", "--csv", required=False, action='store_true', help="path to csv file with processed dataset")
args = vars(ap.parse_args())

if args.get("dataset") is None and args.get("csv") is None:
    print ("User must provide one of the following optionals args: --dataset or --csv ")
    sys.exit(1)

if args.get("dataset"):
    data = []
    labels = []

    print ("[INFO] describing images...")
    imagePaths = list(paths.list_images(args["dataset"]))
    for i, imagePath in enumerate(imagePaths):
        image = cv2.imread(imagePath)
        label = imagePath.split(os.path.sep)[-1].split(".")[0]

        features = image_to_feature_vector(image)
        data.append(features)
        labels.append(label)

        if i > 0 and i % 1000 == 0:
            print ("[INFO] processed {}/{}".format(i, len(imagePaths)))

    le = LabelEncoder()
    labels = le.fit_transform(labels)

    data = np.array(data) / 255.0
    labels = np_utils.to_categorical(labels, 2)

    print ("[INFO] saving processed data into csv files...")
    np.savetxt("data/processed_data.csv", data, delimiter=";")
    np.savetxt("data/processed_labes.csv", labels, delimiter=";")

elif args.get("csv"):
    print ("[INFO] loading processed data from csv files...")
    data = pd.read_csv("data/processed_data.csv", delimiter=";", header=None).values
    labels = pd.read_csv("data/processed_labes.csv", delimiter=";", header=None).values

print ("[INFO] constructing training/testing split...")
trainData, testData, trainLabels, testLabels = train_test_split(data, labels, test_size=0.25, random_state=42)

# 3072-768-384-2 feedforward neural network
# 32 x 32 x 3 = 3,072 

model = Sequential()
model.add(Dense(768, input_dim=3072, init="uniform", activation="relu"))
model.add(Dense(384, init="uniform", activation="relu"))
model.add(Dense(2))
model.add(Activation("softmax"))

print ("[INFO] compiling model...")
sgd = SGD(lr=0.01)
model.compile(loss="binary_crossentropy", optimizer=sgd, metrics=["accuracy"])
model.fit(trainData, trainLabels, nb_epoch=50, batch_size=128)

print ("[INFO] evaluating on testing set...")
loss, accuracy = model.evaluate(testData, testLabels, batch_size=128, verbose=1)
print ("[INFO] loss = {:.4f}, accuracy = {:.4f}%".format(loss, accuracy * 100))