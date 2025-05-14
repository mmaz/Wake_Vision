# %%
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
# %%
# pip install -e . failed
import sys
pardir = str(Path(__file__).resolve().parent.parent)
if pardir not in sys.path:
    sys.path.append(str(pardir))
from wake_vision_loader import get_wake_vision
from experiment_config import default_cfg
from experiments.car_cfg_def import get_car_cfg

# %%
boxable_class_url = "https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions-boxable.csv"
boxable_class_df = pd.read_csv(boxable_class_url)
print(boxable_class_df.shape) 
# target_name = "Car"
target_name = "Bird"
label_name = boxable_class_df[boxable_class_df["DisplayName"] == target_name]["LabelName"].values[0]
print(label_name)
# LabelName /m/0k4j for DisplayName "Car"
# LabelName /m/015p6 for DisplayName "Bird"
#
# %%
# import os
# get_wake_vision expects cleaned_csvs to be in current working directory
# os.chdir(Path(__file__).resolve().parent.parent)
# os.getcwd()
# %%
cfg = default_cfg
cfg.WV_DIR = "/n/netscratch/janapa_reddi_lab/Lab/mmaz/openimages/"

builder = tfds.builder(
    "partial_open_images_v7",
    data_dir=cfg.WV_DIR,
)
# see experiment_config.py, image-level labels:
assert builder.info.features["objects"]["label"].str2int("/m/01g317") == 14048
# boxable car:
car_label_name = "/m/0k4j"
print(builder.info.features["bobjects"]["label"].str2int(car_label_name))
# 570
bird_label_name = "/m/015p6"
print(builder.info.features["bobjects"]["label"].str2int(bird_label_name))
# 21
# %%

# %%
car_cfg = get_car_cfg()
train, val, test = ds = get_wake_vision(car_cfg)


# %%
# test dataset size
#rebatch to 1
test = test.unbatch().batch(1)
# %%
# sample images from test
# %%
for image,label in test.skip(104).take(1):
    if tf.reduce_min(image) < 0:
        image = (image + 1) / 2 
    # print(image.dtype, "--", tf.reduce_min(image), "---", tf.reduce_max(image))
    print(image.shape, label.shape)
    plt.imshow(image[0])
    plt.show()
    print("label", label.numpy()[0])
# %%