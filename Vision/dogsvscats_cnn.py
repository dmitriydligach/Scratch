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

  data_augmentation = keras.Sequential([
      layers.RandomFlip("horizontal"),
      layers.RandomRotation(0.1),
      layers.RandomZoom(0.2)])

  inputs = keras.Input(shape=(180, 180, 3))

  x = data_augmentation(inputs)
  x = layers.Rescaling(1. / 255)(x)

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
  x = layers.Dropout(0.5)(x)

  outputs = layers.Dense(1, activation="sigmoid")(x)

  model = keras.Model(inputs=inputs, outputs=outputs)

  return model

def data_augmentation_example():
  """This is augmenting"""

  train_dataset, validation_dataset, test_dataset = get_data()

  data_augmentation = keras.Sequential([
      layers.RandomFlip("horizontal"),
      layers.RandomRotation(0.1),
      layers.RandomZoom(0.2)])

  plt.figure(figsize=(10, 10))
  for images, _ in train_dataset.take(1):
      for i in range(9):
          augmented_images = data_augmentation(images)
          ax = plt.subplot(3, 3, i + 1)
          plt.imshow(augmented_images[0].numpy().astype("uint8"))
          plt.axis("off")

def main():
  """Main street"""

  model = get_model()
  train_dataset, validation_dataset, test_dataset = get_data()

  model.compile(loss="binary_crossentropy",
                optimizer="rmsprop",
                metrics=["accuracy"])
  callbacks = [keras.callbacks.ModelCheckpoint(
    filepath="cnn_with_augmentation.keras",
    save_best_only=True,
    monitor="val_loss")]
  history = model.fit(
    train_dataset,
    epochs=30,
    validation_data=validation_dataset,
    callbacks=callbacks)

  test_model = keras.models.load_model("convnet_from_scratch.keras")
  test_loss, test_acc = test_model.evaluate(test_dataset)
  print(f"Test accuracy: {test_acc:.3f}")

if __name__ == "__main__":

  # data_augmentation_example()

  main()
