import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor
from transformers import AutoModelForImageClassification, ViTImageProcessor

# check point from saved model (will need to update this for HPC)
MODEL_PATH = "./vit-galaxy-results/checkpoint-31440"
model = AutoModelForImageClassification.from_pretrained(MODEL_PATH)
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")

device = "mps" if torch.backends.mps.is_available() else "cpu"
model.to(device)
model.eval()

print("getting data from gz_hubble")
hubble_raw = load_dataset("mwalmsley/gz_hubble", split="train")
print("\t done")

smooth_col = "smooth-or-featured-hubble_smooth_fraction"
feature_col = "smooth-or-featured-hubble_features_fraction"


# > 60% baseline consensus
def filter_high_confidence(example):
    return (example[smooth_col] > 0.6) or (example[feature_col] > 0.6)


hubble_filtered = hubble_raw.filter(filter_high_confidence)
print(f"{len(hubble_filtered)} Hubble images.")

size = processor.size["height"]
normalize = Normalize(
    mean=processor.image_mean,
    std=(
        image_processor.image_std
        if "image_processor" in locals()
        else processor.image_std
    ),
)
eval_transforms = Compose([Resize(size), CenterCrop(size), ToTensor(), normalize])


def transform_fn(examples):
    # PIL image is under "image" key
    examples["pixel_values"] = [
        eval_transforms(img.convert("RGB")) for img in examples["image"]
    ]
    # same 0/1 label as the model training
    examples["label"] = [
        0 if s > f else 1 for s, f in zip(examples[smooth_col], examples[feature_col])
    ]
    return examples


hubble_eval_set = hubble_filtered.with_transform(transform_fn)

all_preds = []
all_targets = []
batch_size = 64

for i in range(0, len(hubble_eval_set), batch_size):
    batch_indices = range(i, min(i + batch_size, len(hubble_eval_set)))

    images_tensors = [hubble_eval_set[idx]["pixel_values"] for idx in batch_indices]
    labels = [hubble_eval_set[idx]["label"] for idx in batch_indices]

    inputs = torch.stack(images_tensors).to(device)

    with torch.no_grad():
        outputs = model(pixel_values=inputs)
        preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()

    all_preds.extend(preds)
    all_targets.extend(labels)

print(f"Accuracy: {accuracy_score(all_targets, all_preds) * 100:.2f}%")
print(
    classification_report(all_targets, all_preds, target_names=["Smooth", "Featured"])
)
