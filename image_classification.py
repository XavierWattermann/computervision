import os

import pandas as pd
import torch
from datasets import Dataset
from PIL import Image
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor
from transformers import (
    Trainer,
    TrainingArguments,
    ViTForImageClassification,
    ViTImageProcessor,
)


def galaxy_dataset():
    """
    Filters the dataset based on the labesl

    need to have the images downloaded locally, should use HF for this instead / when
    using hpc
    """
    # TODO: these should not be hardcoded -- maybe use hugginface to stream some of this
    IMAGE_DIR = "/Users/xavierwattermann/Desktop/archive/images_gz2/images"
    KAGGLE_CSV = "/Users/xavierwattermann/Desktop/archive/gz2_filename_mapping.csv"
    OFFICIAL_DATA = "/Users/xavierwattermann/Downloads/zoo2MainSpecz.csv.gz"

    df_kaggle = pd.read_csv(KAGGLE_CSV)
    df_official = pd.read_csv(OFFICIAL_DATA, sep=",")
    df_merged = pd.merge(df_kaggle, df_official, left_on="objid", right_on="dr7objid")

    smooth_col = "t01_smooth_or_features_a01_smooth_fraction"
    feature_col = "t01_smooth_or_features_a02_features_or_disk_fraction"

    # Filter for consensus (> 60% agreement)
    df_clean = df_merged[
        (df_merged[smooth_col] > 0.6) | (df_merged[feature_col] > 0.6)
    ].copy()

    # 0 = Smooth/Elliptical, 1 = Featured/Spiral
    df_clean["label"] = df_clean.apply(
        lambda r: 0 if r[smooth_col] > r[feature_col] else 1, axis=1
    )

    df_clean["image_path"] = df_clean["asset_id"].apply(
        lambda x: os.path.join(IMAGE_DIR, f"{int(x)}.jpg")
    )
    df_clean = df_clean[df_clean["image_path"].apply(os.path.exists)]
    return df_clean


df_clean = galaxy_dataset()
data_dict = {
    "image_path": df_clean["image_path"].tolist(),
    "label": df_clean["label"].tolist(),
}
raw_dataset = Dataset.from_dict(data_dict)

# 80% / 20% split
split_dataset = raw_dataset.train_test_split(test_size=0.2)
train_dataset = split_dataset["train"]
val_dataset = split_dataset["test"]

# TODO: add link to this in report
checkpoint = "google/vit-base-patch16-224"
image_processor = ViTImageProcessor.from_pretrained(checkpoint)

size = image_processor.size["height"]
normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
_transforms = Compose([Resize(size), CenterCrop(size), ToTensor(), normalize])


def transform_fn(examples):

    examples["pixel_values"] = [
        _transforms(Image.open(path).convert("RGB")) for path in examples["image_path"]
    ]
    return examples


train_dataset = train_dataset.with_transform(transform_fn)
val_dataset = val_dataset.with_transform(transform_fn)

model = ViTForImageClassification.from_pretrained(
    checkpoint,
    num_labels=2,
    ignore_mismatched_sizes=True,
)

device_arg = "mps" if torch.backends.mps.is_available() else "cpu"

training_args = TrainingArguments(
    output_dir="./vit-galaxy-results",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    # TODO: some of these be tweaked
    eval_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    load_best_model_at_end=True,
    # ------------------------------------------------------
    logging_steps=10,
    learning_rate=5e-5,
    remove_unused_columns=False,
    num_train_epochs=3,
    fp16=False,
    # change this for HPC?
    use_cpu=(device_arg == "cpu"),
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# train the model, comment out if needed to run from check 310XX whatever it is
trainer.train()
