import matplotlib.pyplot as plt
from pathlib import Path
import tensorflow_datasets as tfds

import sys

pardir = str(Path(__file__).resolve().parent.parent)
if pardir not in sys.path:
    sys.path.append(str(pardir))
from wake_vision_loader import get_wake_vision
from experiment_config import default_cfg
from experiments.car_cfg_def import get_car_cfg
import fire
import tensorflow as tf
import tqdm
import numpy as np


def get_car_ds():
    car_cfg = get_car_cfg()
    train, val, test = get_wake_vision(car_cfg)
    return dict(train=train, val=val, test=test)


def save_car_images(
    n_to_skip=5_000,
    n_to_sample=5_000,
    n_to_save=500,
    save_dir=Path(
        "/n/netscratch/janapa_reddi_lab/Lab/mmaz/holy/astro205/labelstudio_images/wakevision_cars"
    ),
    split="test",
    seed=0,
):
    """
    this skips forward n_to_skip images (e.g., if test has 24K images and
    skip=5K, this will iterate starting from 5K), then it draws n_to_sample (5K,
    i.e., images 5000-10000) and randomly subsamples n_to_save (500) images from
    this draw.
    """
    assert save_dir.is_dir(), f"{save_dir=} is not a directory"
    assert len(list(save_dir.iterdir())) == 0, f"{save_dir=} is not empty"

    ds = get_car_ds()[split]
    ds = ds.unbatch().batch(1)
    ds = ds.skip(n_to_skip).take(n_to_sample)

    rng = np.random.default_rng(seed)
    ixs_to_sample = set(rng.choice(n_to_sample, n_to_save, replace=False))
    for ix, (image, label) in tqdm.tqdm(
        enumerate(ds), total=n_to_sample, desc=f"Saving {split} images"
    ):
        if ix not in ixs_to_sample:
            continue
        label_id = label.numpy()[0]
        target_fn = save_dir / f"{split}_ix_{ix:05d}_label_{label_id}.jpg"
        # Rescale from [-1, 1] to [0, 255]
        image = tf.squeeze(image, axis=0)
        image = (image + 1.0) * 127.5
        image = tf.cast(image, tf.uint8)
        encoded_image = tf.image.encode_png(image)
        tf.io.write_file(str(target_fn), encoded_image)


def get_car_sizes():
    test = get_car_ds()["test"]
    # test dataset size
    # rebatch to 1
    test = test.unbatch().batch(1)
    # get dataset size
    test_size = 0
    for _ in test:
        test_size += 1
    print("test size", test_size)  # 24_476


if __name__ == "__main__":
    fire.Fire(save_car_images)
