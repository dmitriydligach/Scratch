#!/usr/bin/env python3

import matplotlib.pyplot as plt
import os, shutil, pathlib
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import image_dataset_from_directory

new_base_path = './CatsVsDogsSmall/'

def get_data():
  """Preprocess data into floating-point tensors for training"""

  new_base_dir = pathlib.Path(new_base_path)

  train_dataset = image_dataset_from_directory(
    new_base_dir / "train",
    image_size=(180, 180),
    batch_size=32)
  validation_dataset = image_dataset_from_directory(
    new_base_dir / "validation",
    image_size=(180, 180),
    batch_size=32)
  test_dataset = image_dataset_from_directory(
    new_base_dir / "test",
    image_size=(180, 180),
    batch_size=32)

  return train_dataset, validation_dataset, test_dataset

def get_model():
  """Get us on CNN"""

  inputs = keras.Input(shape=(180, 180, 3))

  x = layers.Rescaling(1. / 255)(inputs)
  x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(x)
  x = layers.MaxPooling2D(pool_size=2)(x)
  x = layers.Conv2D(filters=64, kernel_size=3, activation="relu")(x)
  x = layers.MaxPooling2D(pool_size=2)(x)
  x = layers.Conv2D(filters=128, kernel_size=3, activation="relu")(x)
  x = layers.MaxPooling2D(pool_size=2)(x)
  x = layers.Conv2D(filters=256, kernel_size=3, activation="relu")(x)
  x = layers.MaxPooling2D(pool_size=2)(x)
  x = layers.Conv2D(filters=256, kernel_size=3, activation="relu")(x)
  x = layers.Flatten()(x)

  outputs = layers.Dense(1, activation="sigmoid")(x)

  model = keras.Model(inputs=inputs, outputs=outputs)

  return model

def main():
  """Main street"""

  model = get_model()

  model.compile(loss="binary_crossentropy",
                optimizer="rmsprop",
                metrics=["accuracy"])

  train_dataset, validation_dataset, test_dataset = get_data()

if __name__ == "__main__":

  main()