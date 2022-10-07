#!/usr/bin/env python3

from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import os, shutil, pathlib

orig_data_path = '/Users/Dima/Work/Data/DogsVsCatsKaggle/train'
new_data_path = './CatsVsDogsSmall/'

def make_subset(subset_name, start_index, end_index):
  """Subsets"""

  for category in ("cat", "dog"):
    dir = new_base_dir / subset_name / category
    os.makedirs(dir)
    fnames = [f"{category}.{i}.jpg"
              for i in range(start_index, end_index)]
    for fname in fnames:
      shutil.copyfile(src=original_dir / fname,
                      dst=dir / fname)

def make_data_dir():
  """Reformat data for experiments"""

original_dir = pathlib.Path(orig_data_path)
new_base_dir = pathlib.Path(new_data_path)

make_subset("train", start_index=0, end_index=1000)
make_subset("validation", start_index=1000, end_index=1500)
make_subset("test", start_index=1500, end_index=2500)

if __name__ == "__main__":

  make_data_dir()