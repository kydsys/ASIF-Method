import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torch.nn.functional import softmax
from torchvision.models import resnext50_32x4d


# =========================
# Basic Configuration
# =========================
batch_size = 128
num_worker = 6
GPU = "cuda:" + "0"
device = torch.device(GPU if torch.cuda.is_available() else "cpu")
print("device", device)


confidence_threshold = 0.95

data_transforms = {
    "test": transforms.Compose(
        [
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    ),
}


# =========================
# Dataset
# =========================
class CreateDatasetFromImages(Dataset):
    def __init__(self, csv_path, file_path, transform=None, label_map_path=None):
        self.file_path = file_path
        self.data_info = pd.read_csv(csv_path)
        self.image_arr = self.data_info["filename"].astype(str).values
        self.label_arr = self.data_info["label"].astype(str).values
        self.transform = transform

        if not (label_map_path and os.path.exists(label_map_path)):
            raise RuntimeError(f"label_to_index.json not found: {label_map_path}")

        with open(label_map_path, "r") as f:
            self.label_to_index = json.load(f)

    def __getitem__(self, index):
        single_image_name = self.image_arr[index]
        single_image_path = os.path.join(self.file_path, single_image_name)
        img_as_img = Image.open(single_image_path).convert("RGB")

        if self.transform:
            img_as_img = self.transform(img_as_img)

        label = self.label_arr[index]
        label_str = str(label)
        if label_str not in self.label_to_index:
            raise RuntimeError(f"Label '{label_str}' not found in label map.")
        label_index = int(self.label_to_index[label_str])

        return img_as_img, torch.tensor(label_index, dtype=torch.long), single_image_name, label_str

    def __len__(self):
        return len(self.image_arr)


# =========================
# Step folder
# =========================
def get_step_folders():
    current_dir = os.getcwd()
    step_folders = [
        os.path.join(current_dir, folder)
        for folder in os.listdir(current_dir)
        if folder.startswith("Step") and os.path.isdir(os.path.join(current_dir, folder))
    ]
    return sorted(step_folders)


# =========================
# load label map
# =========================
def load_index_to_label(label_map_path: str):
    with open(label_map_path, "r") as f:
        label_map = json.load(f)
    return {int(v): k for k, v in label_map.items()}


# =========================
# model load
# =========================
def load_model(step_folder: str, num_classes: int, device: torch.device):
    model_path = os.path.join(step_folder, "model.pth")
    if not os.path.exists(model_path):
        raise RuntimeError(f"model.pth not found: {model_path}")

    model = resnext50_32x4d(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


# =========================
# load T（model_T.json）
# =========================
def load_trained_temperature(step_folder: str) -> float:
    temp_path = os.path.join(step_folder, "model_T.json")
    if not os.path.exists(temp_path):
        raise RuntimeError(f"Temperature file not found: {temp_path}")

    with open(temp_path, "r") as f:
        obj = json.load(f)

    T = float(obj["temperature"])
    if not np.isfinite(T) or T <= 0:
        raise RuntimeError(f"Invalid temperature in {temp_path}: {T}")

    print(f"[LOAD] Temperature -> {T:.6f}")
    return T


# =========================
# read allowlist（integrated_f1_scores.csv）
# =========================
def load_allowlist(allowlist_csv: str) -> set:
    df = pd.read_csv(allowlist_csv)
    if "Species" not in df.columns:
        raise RuntimeError(f"{allowlist_csv} must contain column 'Species'")

    wl = (
        df["Species"]
        .dropna()
        .astype(str)
        .str.strip()
        .replace("", np.nan)
        .dropna()
        .unique()
        .tolist()
    )
    return set(wl)   # set[str]


# =========================
# Predicted
# =========================
def predict_test_data(
    model,
    test_loader,
    results_csv,
    label_map_path,
    species_above_threshold,
    temperature,
    confidence_threshold,
):
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []
    all_filenames = []
    all_probs = []

    if os.path.exists(label_map_path):
        with open(label_map_path, "r") as f:
            index_to_label = {int(v): k for k, v in json.load(f).items()}
    else:
        raise RuntimeError(f"label_map not found: {label_map_path}")

    with torch.no_grad():
        val_bar = tqdm(test_loader, desc="Evaluating (scaled)")
        for inputs, label_index, filenames, actual_labels in val_bar:
            inputs = inputs.to(device)

            logits = model(inputs)
            outputs = logits / float(temperature)
            probabilities = softmax(outputs, dim=1)

            _, predicted = torch.max(outputs, 1)
            predicted_labels = [index_to_label[int(idx)] for idx in predicted.cpu().numpy()]

            all_preds.extend(predicted_labels)
            all_labels.extend(actual_labels)
            all_filenames.extend(filenames)
            all_probs.extend(probabilities.cpu().numpy())

    probs_array = np.array(all_probs)
    probs_df = pd.DataFrame(
        probs_array, columns=[f"Prob_{i}" for i in range(probs_array.shape[1])]
    ).round(6)

    result = pd.DataFrame(
        {
            "filename": all_filenames,
            "Actual": all_labels,
            "Predicted": all_preds,
        }
    )
    result = pd.concat([result, probs_df], axis=1)

    os.makedirs("raw", exist_ok=True)
    result.to_csv(os.path.join("raw", f"test_all_scaled.csv"), index=False)

    probability_columns = result.columns[3:]
    result[probability_columns] = result[probability_columns].astype(float)
    max_prob = result[probability_columns].max(axis=1)

    mask_above = (
        result["Predicted"].isin(species_above_threshold)
        & (max_prob >= float(confidence_threshold))
    )

    df_results = result.loc[mask_above, ["filename", "Actual", "Predicted"]]
    df_results.to_csv(results_csv, mode="a", header=False, index=False)

    df_low = result.loc[~mask_above, ["filename", "Actual", "Predicted"]]
    df_low.to_csv("low_config.csv", mode="a", header=False, index=False)


# =========================
# main
# =========================
def main():
    img_path = ""
    test_csv = ""
    
    step_folder = "Step1"
    print("="*60 + f"\nNow is Processing test csv...\nconfidence_threshold:{confidence_threshold}\nTEST CSV: {test_csv}\n" + "="*60)

    label_map_path = os.path.join(step_folder, "label_to_index.json")
    index_to_label = load_index_to_label(label_map_path)
    if not index_to_label:
        raise RuntimeError(f"label_to_index.json empty: {label_map_path}")

    allowlist_csv = "integrated_f1_scores.csv"
    allowlist = load_allowlist(allowlist_csv)

    print(f"Allow list species (n={len(allowlist)})\n" + "-"*60 + "\n" + "\n".join(f"- {s}" for s in sorted(allowlist)) + "\n" + "-"*60)

    test_data = CreateDatasetFromImages(
        csv_path=test_csv,
        file_path=img_path,
        transform=data_transforms["test"],
        label_map_path=label_map_path,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_worker,
        pin_memory=True,
        persistent_workers=True,
    )

    num_classes = len(index_to_label)
    model = load_model(step_folder, num_classes, device)

    T = load_trained_temperature(step_folder)

    results_csv = "Results.csv"
    with open(results_csv, "w") as f:
        f.write("filename,Actual,Predicted\n")

    with open("low_config.csv", "w") as f:
        f.write("filename,Actual,Predicted\n")

    predict_test_data(
        model=model,
        test_loader=test_loader,
        results_csv=results_csv,
        label_map_path=label_map_path,
        species_above_threshold=allowlist,
        temperature=T,
        confidence_threshold=confidence_threshold,
    )


    total_count = len(pd.read_csv(test_csv))
    auto_count = len(pd.read_csv("Results.csv"))
    print(f"auto_Rare: {auto_count / total_count * 100:.2f}%")


if __name__ == "__main__":
    main()
