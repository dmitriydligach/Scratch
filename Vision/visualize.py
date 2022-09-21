#!/usr/bin/env python3

from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

if __name__ == "__main__":

  (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

  # plot digit
  plt.imshow(train_images[111], cmap=plt.cm.binary);
  plt.show()

  # show pixel values
  print(train_images[111])