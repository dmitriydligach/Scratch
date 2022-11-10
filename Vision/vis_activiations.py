#!/usr/bin/env python3

from tensorflow.keras import layers
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

new_base_path = './CatsVsDogsSmall/'

from tensorflow import keras
import numpy as np

img_path = keras.utils.get_file(
  fname="cat.jpg",
  origin="https://img-datasets.s3.amazonaws.com/cat.jpg")

def show_image(img_path):
  """Show yourself"""

  img_tensor = get_img_array(img_path, target_size=(180, 180))
  plt.axis("off")
  plt.imshow(img_tensor[0].astype("uint8"))
  plt.show()

def get_img_array(img_path, target_size):
  """Imagine there's no hunger"""

  img = keras.utils.load_img(
    img_path, target_size=target_size)
  array = keras.utils.img_to_array(img)
  array = np.expand_dims(array, axis=0)

  return array

def get_layer_activations(model, img_tensor):
  """Activate my layers"""

  layer_outputs = []
  layer_names = []

  for layer in model.layers:
    if isinstance(layer, (layers.Conv2D, layers.MaxPooling2D)):
      layer_outputs.append(layer.output)
      layer_names.append(layer.name)

  activation_model = keras.Model(inputs=model.input, outputs=layer_outputs)
  activations = activation_model.predict(img_tensor)

  return activations

def main():
  """My main"""

  model = model = keras.models.load_model('cnn_with_augmentation.keras')
  img_tensor = get_img_array(img_path, target_size=(180, 180))
  activations = get_layer_activations(model, img_tensor)

  # it's a 178 × 178 feature map with 32 channels
  first_layer_activation = activations[0]
  print(first_layer_activation.shape)

  # plot 5th channel
  plt.matshow(first_layer_activation[0, :, :, 5], cmap="viridis")
  plt.show()

if __name__ == "__main__":

  main()