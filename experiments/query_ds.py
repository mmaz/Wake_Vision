import matplotlib.pyplot as plt
from pathlib import Path
import tensorflow_datasets as tfds
import collections
import json

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
import sklearn.metrics


def get_car_ds():
    car_cfg = get_car_cfg()
    train, val, test = get_wake_vision(car_cfg)
    return dict(train=train, val=val, test=test)


def get_bird_ds():
    # modify the car_cfg to use bird
    cfg = get_car_cfg()
    cfg.BBOX_PERSON_DICTIONARY = {"Bird": 21}
    train, val, test = get_wake_vision(cfg)
    return dict(train=train, val=val, test=test)


def save_images(
    target: str,
    n_to_save=500,
    split="test",
    seed=0,
):
    """
    this skips forward n_to_skip images (e.g., if test has 24K images and
    skip=5K, this will iterate starting from 5K), then it draws n_to_sample (5K,
    i.e., images 5000-10000) and randomly subsamples n_to_save (500) images from
    this draw.
    """
    save_parent = Path(
        "/n/netscratch/janapa_reddi_lab/Lab/mmaz/holy/astro205/labelstudio_images"
    )
    assert target in ["car", "bird"]
    if target == "car":
        save_dir = save_parent / "wakevision_cars"
        ds = get_car_ds()[split]
        n_to_skip = (5_000,)
        n_to_sample = (5_000,)
    elif target == "bird":
        save_dir = save_parent / "wakevision_birds"
        ds = get_bird_ds()[split]
        n_to_skip = 500
        n_to_sample = 1000
    print(f"{n_to_skip=}, {n_to_sample=}")
    assert save_dir.is_dir(), f"{save_dir=} is not a directory"
    assert len(list(save_dir.iterdir())) == 0, f"{save_dir=} is not empty"

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


def report_tfds_balance(
    target: str,
):
    if target == "car":
        image_dir: Path = Path(
            "/n/netscratch/janapa_reddi_lab/Lab/mmaz/holy/astro205/labelstudio_images/wakevision_cars"
        )
    elif target == "bird":
        image_dir: Path = Path(
            "/n/netscratch/janapa_reddi_lab/Lab/mmaz/holy/astro205/labelstudio_images/wakevision_birds"
        )
    """
    report balance of dataset
    """
    counter = collections.Counter()
    for image_path in image_dir.iterdir():
        # example filename: test_ix_04130_label_1.jpg
        label_id = int(image_path.name.split("_")[-1].split(".")[0])
        counter[label_id] += 1
    print(counter)  # car: Counter({0: 254, 1: 246}), bird: Counter({0: 235, 1: 265})


def human_val_classification_report(
    target: str,
):
    """
    entries are saved in the following format:

      {
        "image": "\/data\/local-files\/?d=wakevision_cars\/test_ix_00024_label_1.jpg",
        "id": 1678,
        "choice": "car",
        "annotator": 1,
        "annotation_id": 649,
        "created_at": "2025-04-28T23:31:40.472616Z",
        "updated_at": "2025-04-28T23:31:40.472649Z",
        "lead_time": 1.341
    },
    {
        "image": "\/data\/local-files\/?d=wakevision_cars\/test_ix_00031_label_0.jpg",
        "id": 1679,
        "choice": "background",
        "annotator": 1,
        "annotation_id": 650,
        "created_at": "2025-04-28T23:31:42.438735Z",
        "updated_at": "2025-04-28T23:31:42.438764Z",
        "lead_time": 1.073
    },

    label_1 indicates that WakeVision considers the image a car, likewise label_0 background
    choice: [car, background] indicates human-assigned label (without knowledge of wakevision's label)

    this function generates a classification report treating human-assigned labels as ground truth
    """
    if target == "car":
        labelstudio_json: Path = (
            Path(
                "/n/holylabs/LABS/janapa_reddi_lab/Users/mmaz/wakevision_work/Wake_Vision/wakevision-at-2025-04-29-00-04-5e08b2ab.json"
            ),
        )
    elif target == "bird":
        raise NotImplementedError("bird not implemented yet")
    label_data = json.loads(labelstudio_json.read_text())
    y_pred = []
    y_true = []
    for entry in label_data:
        # example filename: test_ix_04130_label_1.jpg
        wakevision_label_id = int(entry["image"].split("_")[-1].split(".")[0])
        y_pred.append(wakevision_label_id)
        y_true.append(1 if entry["choice"] == target else 0)

    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    # print confusion matrix
    cm = sklearn.metrics.confusion_matrix(y_true=y_true, y_pred=y_pred)
    print("Confusion matrix")
    print(cm)

    # print classification report
    print(
        sklearn.metrics.classification_report(
            y_true, y_pred, target_names=["background", target]
        )
    )


def get_sizes(target: str, split: str, batch_size: int):
    assert target in ["car", "bird"]
    assert batch_size > 0
    if target == "car":
        ds = get_car_ds()
    elif target == "bird":
        ds = get_bird_ds()
    assert split in ["train", "val", "test"]
    ds = ds[split]
    # need to flush when using tee + tf's stdout behavior
    print(f"{split=} loaded, {batch_size=}", flush=True)
    dataset_size = 0
    # rebatch to BS
    for _ in ds.unbatch().batch(batch_size):
        dataset_size += 1
        if batch_size > 1 and dataset_size % 100 == 0:
            print(f"Calculating {split=} {batch_size*dataset_size=}...", flush=True)
    print(
        f"final size: {split=} {dataset_size=} {dataset_size*batch_size=}", flush=True
    )
    # bird:
    # test: 3008

    # car:
    # test: 24_476
    # split='val' dataset_size=8264
    #


# module load python cuda/12.4.1-fasrc01 cudnn/9.5.1.17_cuda12-fasrc01
# conda activate wakevision_env
if __name__ == "__main__":
    fire.Fire(report_tfds_balance)
