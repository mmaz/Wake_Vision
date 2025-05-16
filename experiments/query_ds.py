import matplotlib.pyplot as plt
from pathlib import Path
import tensorflow_datasets as tfds
import collections
import json
import subprocess
import shutil
import tempfile

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
from PIL import Image
from loguru import logger


def get_car_ds(wv_dir: str | None = None):
    car_cfg = get_car_cfg()
    if wv_dir is not None:
        car_cfg.WV_DIR = wv_dir
    train, val, test = get_wake_vision(car_cfg)
    return dict(train=train, val=val, test=test)


def get_bird_ds(wv_dir: str | None = None):
    # modify the car_cfg to use bird
    cfg = get_car_cfg(wv_dir)
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

def make_collage(
    target: str,
    label_id: int,
    split: str = "test",
    grid_size: tuple[int, int] = (3, 3),
    margin: int = 10,
    seed: int = 0,
    skip: int | tuple[int, ...] | None = None,
    sample: int | None = None,
) -> Image.Image:
    """
    Pulls out grid_size[0] * grid_size[1] random images with label==1
    from the specified dataset slice, and returns a single PIL.Image
    collage with `margin` pixels of white border around each.

    Args:
        target: "car" or "bird"
        split: which split of the dataset ("train"/"val"/"test")
        grid_size: (cols, rows) of the grid, defaults to (3,3)
        margin: pixels of whitespace between (and around) each tile
        seed: random seed for reproducibility
        skip: how many examples to skip before sampling (overrides defaults)
        sample: how many examples to draw before filtering to label==1
    Returns:
        A PIL.Image of size
        (cols*W + (cols+1)*margin) × (rows*H + (rows+1)*margin).
    """
    assert target in ["car", "bird"]
    if target == "car":
        ds = get_car_ds()[split]
        default_skip, default_sample = 5_000, 30
    else:
        ds = get_bird_ds()[split]
        default_skip, default_sample = 500, 30

    n_skip = skip if skip is not None else default_skip
    n_sample = sample if sample is not None else default_sample

    ds = ds.unbatch().skip(n_skip).take(n_sample)

    imgs: list[np.ndarray] = []
    labs: list[np.ndarray] = []
    for image, label in tfds.as_numpy(ds):
        # image in [-1,1], convert to [0,255]
        im = ((image + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        imgs.append(im)
        labs.append(label.squeeze())

    # filter for label_id
    imgs = [img for img, lab in zip(imgs, labs) if lab == label_id]
    n_needed = grid_size[0] * grid_size[1]
    if len(imgs) < n_needed:
        raise RuntimeError(
            f"Found only {len(imgs)} label-1 images, but need {n_needed}."
        )

    # random pick
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(imgs), size=n_needed, replace=False)
    tiles = [Image.fromarray(imgs[i]) for i in chosen]

    # all tiles same size
    W, H = tiles[0].size
    cols, rows = grid_size

    # compute canvas size
    canvas_w = cols * W + (cols + 1) * margin
    canvas_h = rows * H + (rows + 1) * margin
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(255, 255, 255))

    # paste
    for idx, tile in enumerate(tiles):
        row = idx // cols
        col = idx % cols
        x = margin + col * (W + margin)
        y = margin + row * (H + margin)
        canvas.paste(tile, (x, y))

    return canvas


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
        labelstudio_json = Path(
            "/n/holylabs/LABS/janapa_reddi_lab/Users/mmaz/wakevision_work/Wake_Vision/wakevision-at-2025-04-29-00-04-5e08b2ab.json"
        )
    elif target == "birds":
        labelstudio_json = Path(
            "/n/holylabs/LABS/janapa_reddi_lab/Users/mmaz/wakevision_work/Wake_Vision/wakevision-birds-500-project-3-at-2025-05-06-04-25-d9628515.json"
        )
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


def get_sizes(target: str, split: str, batch_size: int, wv_dir: str | None = None):
    logger.warning("ENSURE YOU DISABLED repeat() IN WAKE_VISION LOADER")
    assert target in ["car", "birds"]
    assert batch_size > 0
    if target == "car":
        ds = get_car_ds(wv_dir=wv_dir)
    elif target == "birds":
        ds = get_bird_ds(wv_dir=wv_dir)
    assert split in ["train", "val", "test"]
    ds = ds[split]
    logger.info(f"{split=} loaded, {batch_size=}")
    batch_count = 0
    positives_count = 0
    backgrounds_count = 0
    # rebatch to BS
    for images, labels in ds.unbatch().batch(batch_size):
        batch_count += 1
        labels = tf.squeeze(labels)
        positives_count += tf.reduce_sum(labels).numpy()
        backgrounds_count += batch_size - tf.reduce_sum(labels).numpy()
        if batch_size > 1 and batch_count % 100 == 0:
            logger.debug(
                f"Calculating {split=} {batch_size*batch_count=} {positives_count=} {backgrounds_count=}..."
            )
    logger.info(f"RESULTS: {target=} final size: {split=} {batch_count=} {batch_count*batch_size=} {positives_count=} {backgrounds_count=}")  # fmt: skip
    # bird:
    # test: 3008

    # car:
    # test: 24_476
    # split='val' dataset_size=8264
    #

def get_sizes_bs1(target: str, split: str):
    assert target in ["car", "birds"]
    if target == "car":
        ds = get_car_ds()
    elif target == "birds":
        ds = get_bird_ds()
    assert split in ["train", "val", "test"]
    ds = ds[split]
    logger.info(f"{split=} loaded")
    count = 0
    positives_count = 0
    backgrounds_count = 0
    for image, label in ds.unbatch().batch(1):
        count += 1
        if label.numpy() == 1:
            positives_count += 1
        else:
            backgrounds_count += 1
        if count % 500 == 0:
            logger.debug(
                f"Calculating {split=} {count=} {positives_count=} {backgrounds_count=}..."
            )
    logger.info(f"RESULTS: {target=} final size: {split=} {count=} {positives_count=} {backgrounds_count=}")  # fmt: skip

def copy_oi_to_ram() -> tempfile.TemporaryDirectory:
    """
    we unpack the openimages tar to /dev/shm
    - tar xf will create openimages/1.0.0/
    - we need /dev/shm/tempdir/partial_open_images_v7/1.0.0/
    - we will return /dev/shm/tempdir as cfg.WV_DIR

    note most scratch partitions have at most 396GB of space
    https://docs.rc.fas.harvard.edu/kb/running-jobs/#Slurm_partitions
    https://docs.rc.fas.harvard.edu/kb/policy-scratch/
    shm can be made large enough by the --mem flag
    """
    src_oi_tar = Path("/n/netscratch/janapa_reddi_lab/Lab/mmaz/openimages.tar")
    assert src_oi_tar.is_file(), f"{src_oi_tar} is not a file"
    tmpdir = tempfile.TemporaryDirectory(dir=Path("/dev/shm/"))
    logger.info(f"Unpacking {src_oi_tar} to {tmpdir.name}")
    # untar:
    # -C change to directory tmpdir.name
    cmd = f"tar xf {src_oi_tar} -C {tmpdir.name}"
    logger.info(f"Running command: {cmd}")
    subprocess.run(cmd, shell=True, check=True)
    # rename openimages/1.0.0/ to partial_open_images_v7/1.0.0/
    shutil.move(
        Path(tmpdir.name) / "openimages",
        Path(tmpdir.name) / "partial_open_images_v7",
    )
    assert (Path(tmpdir.name) / "partial_open_images_v7/1.0.0").is_dir()
    logger.info("Unpacking done")
    return tmpdir


# def get_sizes_fasrc_ramdisk(target: str):
def get_sizes_fasrc_ramdisk():
    """alloc at least 512GB ram"""
    logger.add(f"fasrc_ramdisk_pn.log", colorize=True)

    # will be deleted when GC collects it
    wv_dir_td = copy_oi_to_ram()

    batch_size = 1024
    for split in ["test", "val", "train"]:
        # get_sizes(target, split, batch_size, wv_dir=wv_dir_td.name)
        for target in ["car", "birds"]:
            logger.info(f"Calculating {target=} {split=}")
            get_sizes(target, split, batch_size, wv_dir=wv_dir_td.name)

def get_sizes_fasrc_ondisk():
    """128 cores & 128 GB ram"""
    logger.add(f"fasrc_testval_exact_ondisk_pn.log", colorize=True)

    # batch_size = 1024
    # for split in ["test", "val", "train"]:
    for split in ["test", "val"]:
        for target in ["car", "birds"]:
            logger.info(f"Calculating {target=} {split=}")
            # get_sizes(target, split, batch_size)
            get_sizes_bs1(target, split)


def save_collages():
    bird_collage = make_collage(
        target="bird",
        label_id=1,
        split="test",
        grid_size=(3, 3),
        margin=10,
        seed=0,
    )
    car_collage = make_collage(
        target="car",
        label_id=1,
        split="test",
        grid_size=(3, 3),
        margin=10,
        skip=2_000,
        seed=0,
    )
    bird_collage.save("bird_collage.png")
    car_collage.save("car_collage.png")


# module load python cuda/12.4.1-fasrc01 cudnn/9.5.1.17_cuda12-fasrc01
# conda activate wakevision_env
if __name__ == "__main__":
    fire.Fire(get_sizes_fasrc_ondisk)