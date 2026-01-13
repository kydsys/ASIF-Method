import os
import json
import random
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.functional import softmax
from torchvision import transforms
from torchvision.models import resnext50_32x4d, ResNeXt50_32X4D_Weights
from torch.optim.lr_scheduler import LambdaLR, StepLR
from tqdm import tqdm

# ============================================================
# ✅ Global Configuration
# ============================================================
batch_size = 128
num_worker = 6
GPU = "cuda:" + "0"
device = torch.device(GPU if torch.cuda.is_available() else "cpu")
print("device", device)

Recall_Precision = 95
confidence_threshold = 0.95
coverage_threshold = 0.50


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


set_seed(66)

# =========================
# transforms
# =========================
data_transforms = {
    "train": transforms.Compose(
        [
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    ),
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

    def __init__(self, csv_path, file_path, transform=None, label_map_path=None, save_map=False):
        self.file_path = file_path
        self.data_info = pd.read_csv(csv_path)

        self.image_arr = self.data_info["filename"].astype(str).values
        self.label_arr = self.data_info["label"].astype(str).values
        self.transform = transform

        if label_map_path and os.path.exists(label_map_path):
            with open(label_map_path, "r") as f:
                self.label_to_index = json.load(f)
        else:
            unique_labels = np.unique(self.label_arr)
            self.label_to_index = {str(label): int(idx) for idx, label in enumerate(unique_labels)}

            if save_map and label_map_path:
                with open(label_map_path, "w") as f:
                    json.dump(self.label_to_index, f, ensure_ascii=False, indent=2)

    def __getitem__(self, index):
        single_image_name = self.image_arr[index]
        single_image_path = os.path.join(self.file_path, single_image_name)
        img_as_img = Image.open(single_image_path).convert("RGB")

        if self.transform:
            img_as_img = self.transform(img_as_img)

        label = str(self.label_arr[index])
        label_index = torch.tensor(self.label_to_index[label], dtype=torch.long)

        return img_as_img, label_index, single_image_name

    def __len__(self):
        return len(self.image_arr)


# =========================
# Temperature scaling
# =========================
def calibrate_temperature(model, val_loader, device):
    model.eval()
    nll_criterion = nn.CrossEntropyLoss().to(device)

    logits_list, labels_list = [], []
    with torch.no_grad():
        for inputs, labels, _ in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logits = model(inputs)
            logits_list.append(logits)
            labels_list.append(labels)

    logits = torch.cat(logits_list, dim=0)
    labels = torch.cat(labels_list, dim=0)

    T = nn.Parameter(torch.ones(1, device=device) * 1.0)
    optimizer = torch.optim.LBFGS([T], lr=0.01, max_iter=50)

    def closure():
        optimizer.zero_grad()
        loss = nll_criterion(logits / T, labels)
        loss.backward()
        return loss

    optimizer.step(closure)

    temperature = float(T.detach().item())
    print(f"Calibrated temperature (init=1.0): {temperature:.4f}")
    return temperature


# =========================
# validate acc
# =========================
def validate_model(model, val_loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels, _ in tqdm(val_loader, desc="Validating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100.0 * correct / max(1, total)


# =========================
# train + calibrate T
# =========================
def train_model(train_loader, val_loader, model_path, num_classes, patience=7):
    model = resnext50_32x4d(weights=ResNeXt50_32X4D_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)

    warmup_epochs = 5
    total_epochs = 60
    warmup_factor = lambda epoch: epoch / warmup_epochs if epoch <= warmup_epochs else 1
    scheduler_warmup = LambdaLR(optimizer, lr_lambda=warmup_factor)
    scheduler_step = StepLR(optimizer, step_size=10, gamma=0.1)

    best_acc = 0.0
    no_improve = 0

    for epoch in range(total_epochs):
        model.train()
        train_bar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")
        for inputs, labels, _ in train_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        val_acc = validate_model(model, val_loader, device)
        print(f"Epoch {epoch + 1}: Validation Accuracy = {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            no_improve = 0
            torch.save(model.state_dict(), model_path)
            print(f"New best model saved at {model_path}")
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Early stopping triggered.")
                break

        if epoch <= warmup_epochs:
            scheduler_warmup.step()
        else:
            scheduler_step.step()

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    temperature = calibrate_temperature(model, val_loader, device)

    temp_path = model_path.replace(".pth", "_T.json")
    with open(temp_path, "w") as f:
        json.dump({"temperature": temperature}, f, indent=2)

    return model, temperature


# =========================
# inference：return results 、 results_t
# =========================
def predict_val_data(model, val_loader, label_map_path, temperature):
    model.eval()
    model.to(device)

    with open(label_map_path, "r") as f:
        index_to_label = {int(v): k for k, v in json.load(f).items()}

    rows, rows_t = [], []

    with torch.no_grad():
        for inputs, labels, filenames in tqdm(val_loader, desc="Predicting (val)"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            logits = model(inputs)
            probs = softmax(logits, dim=1)
            probs_t = softmax(logits / float(temperature), dim=1)

            pred = probs.argmax(dim=1)
            pred_t = probs_t.argmax(dim=1)

            bs, num_classes = probs.shape

            for i in range(bs):
                actual = index_to_label[int(labels[i].item())]

                row = {
                    "filename": filenames[i],
                    "Actual": actual,
                    "Predicted": index_to_label[int(pred[i].item())],
                }
                row_t = {
                    "filename": filenames[i],
                    "Actual": actual,
                    "Predicted": index_to_label[int(pred_t[i].item())],
                }

                for c in range(num_classes):
                    row[f"Prob_{c}"] = float(probs[i, c].item())
                    row_t[f"Prob_{c}"] = float(probs_t[i, c].item())

                rows.append(row)
                rows_t.append(row_t)

    results = pd.DataFrame(rows)
    results_t = pd.DataFrame(rows_t)

    prob_cols = sorted(
        [c for c in results.columns if c.startswith("Prob_")],
        key=lambda x: int(x.split("_")[1]),
    )

    results = results[["filename", "Actual", "Predicted"] + prob_cols]
    results_t = results_t[["filename", "Actual", "Predicted"] + prob_cols]

    return results, results_t


# =========================
# filter confidence：results_t.csv -> results_t_filtered.csv
# =========================
def filter_confidence(data_path: str, output_path: str, threshold: float) -> pd.DataFrame:
    df = pd.read_csv(data_path)

    prob_cols = [c for c in df.columns if c.startswith("Prob_")]
    if not prob_cols:
        raise ValueError("filter_confidence: no Prob_* columns found in input csv.")

    df["_maxprob"] = df[prob_cols].max(axis=1)
    df_filt = df[df["_maxprob"] >= float(threshold)].copy()
    df_filt.drop(columns=["_maxprob"], inplace=True)

    df_filt.to_csv(output_path, index=False)
    return df_filt


# =========================
# metrics
# =========================
def calculate_metrics(data_path: str, metrics_path: str):
    df = pd.read_csv(data_path)
    if "Actual" not in df.columns or "Predicted" not in df.columns:
        raise ValueError(f"{data_path} must contain columns: Actual, Predicted")

    species = df["Actual"].unique()
    results = []

    for s in species:
        tp = int(((df["Actual"] == s) & (df["Predicted"] == s)).sum())
        fp = int(((df["Actual"] != s) & (df["Predicted"] == s)).sum())
        fn = int(((df["Actual"] == s) & (df["Predicted"] != s)).sum())
        tn = int(((df["Actual"] != s) & (df["Predicted"] != s)).sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        results.append(
            {
                "Species": s,
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "TN": tn,
                "Recall": round(recall, 4) * 100,
                "Precision": round(precision, 4) * 100,
                "F1-Score": round(f1, 4) * 100,
            }
        )

    out = pd.DataFrame(results).sort_values(by="Species")
    out = out[["Species", "TP", "FP", "FN", "TN", "Recall", "Precision", "F1-Score"]]
    out.to_csv(metrics_path, index=False)


def create_step_folder(step: int):
    step_path = f"Step{step}"
    os.makedirs(step_path, exist_ok=True)
    return step_path


# =========================
# Plot the coverage curve (with threshold line).
# =========================
def plot_gate_coverage_threshold(gate_table: pd.DataFrame, threshold: float, out_png: str):
    if gate_table is None or gate_table.empty:
        raise ValueError("plot_gate_coverage_threshold: gate_table empty")

    y = gate_table["Coverage"].to_numpy(dtype=float)
    x = np.arange(len(y))
    species = gate_table["Species"].astype(str).tolist()

    plt.figure(figsize=(max(10, len(y) * 0.6), 6))
    plt.plot(x, y, marker="o", linewidth=1)

    #Species + Coverage
    for xi, yi, sp in zip(x, y, species):
        plt.text(
            xi, yi,
            f"{sp}\n{yi:.3f}",
            ha="center", va="bottom",
            fontsize=8, rotation=45,
        )

    # ✅ threshold line
    plt.axhline(float(threshold), linestyle="--", color="red")
    plt.title(f"Gate Coverage Curve (Coverage >= {float(threshold):.2f})")
    plt.xlabel("Rank (sorted by Coverage desc)")
    plt.ylabel("Coverage")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=310)
    plt.close()


# =========================
# Gating mechanism（Coverage >= threshold）
# =========================
def gate_from_candidates(results_t, results_t_filtered, candidate_species, output_dir, coverage_threshold=0.50):

    if results_t.empty:
        raise RuntimeError("Gate: results_t is empty.")
    if results_t_filtered.empty:
        raise RuntimeError("Gate: results_t_filtered is empty. Try lower confidence_threshold.")
    if not candidate_species:
        raise RuntimeError("Gate: candidate_species empty.")

    full = results_t.copy()
    full["Actual"] = full["Actual"].astype(str)

    acc = results_t_filtered.copy()
    acc["Actual"] = acc["Actual"].astype(str)

    full_counts = full["Actual"].value_counts().to_dict()
    acc_counts = acc["Actual"].value_counts().to_dict()

    rows = []
    for s in candidate_species:
        s = str(s)
        n_full = int(full_counts.get(s, 0))
        n_acc = int(acc_counts.get(s, 0))
        cov = (n_acc / n_full) if n_full > 0 else 0.0
        rows.append({"Species": s, "N_full_actual": n_full, "N_acc_actual": n_acc, "Coverage": cov})

    gate_table = (
        pd.DataFrame(rows)
        .sort_values(by=["Coverage", "N_full_actual"], ascending=[False, False])
        .reset_index(drop=True)
    )
    gate_table["Coverage_pct"] = (gate_table["Coverage"] * 100.0).round(2)
    gate_table["Pass"] = gate_table["Coverage"] >= float(coverage_threshold)

    # ✅ rule：Coverage >= coverage_threshold
    gate_selected = gate_table[gate_table["Coverage"] >= float(coverage_threshold)].copy().reset_index(drop=True)
    allow_list = pd.DataFrame({"Species": gate_selected["Species"].astype(str).tolist()})

    os.makedirs(output_dir, exist_ok=True)
    gate_table.to_csv(os.path.join(output_dir, "gate_table.csv"), index=False)
    gate_selected.to_csv(os.path.join(output_dir, "gate_selected.csv"), index=False)
    allow_list.to_csv(os.path.join(output_dir, "allow_list.csv"), index=False)

    plot_gate_coverage_threshold(
        gate_table=gate_table,
        threshold=float(coverage_threshold),
        out_png=os.path.join(output_dir, "gate_coverage.png"),
    )

    return allow_list


# =========================
# main
# =========================
def main():
    img_path = ""
    step = 1

    step_path = create_step_folder(step)
    label_map_path = os.path.join(step_path, "label_to_index.json")

    train_csv = ""
    val_csv = ""

    num_classes = pd.read_csv(train_csv)["label"].astype(str).nunique()

    print("="*79 + f"\nconfidence_threshold:{confidence_threshold}\nRecall_Precision:{Recall_Precision}"+f"\ncoverage_threshold:{coverage_threshold}")
    print("="*79 + f"\n▶ ▶ ▶  DATASET  ◀ ◀ ◀\n\nNUM_CLASSES : {num_classes}\nTRAIN CSV   : {train_csv}\nVAL   CSV   : {val_csv}\n" + "="*79)
    
    train_data = CreateDatasetFromImages(
        csv_path=train_csv,
        file_path=img_path,
        transform=data_transforms["train"],
        label_map_path=label_map_path,
        save_map=True,
    )
    val_data = CreateDatasetFromImages(
        csv_path=val_csv,
        file_path=img_path,
        transform=data_transforms["test"],
        label_map_path=label_map_path,
        save_map=False,
    )

    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=num_worker, pin_memory=True, persistent_workers=True,
    )
    val_loader = DataLoader(
        val_data, batch_size=batch_size, shuffle=False, num_workers=num_worker, pin_memory=True,persistent_workers=True,
    )

    model_path = os.path.join(step_path, "model.pth")
    model, temperature = train_model(train_loader, val_loader, model_path, num_classes, patience=7)

    results_csv = os.path.join(step_path, "results.csv")
    results_t_csv = os.path.join(step_path, "results_t.csv")

    results, results_t = predict_val_data(
        model=model,
        val_loader=val_loader,
        label_map_path=label_map_path,
        temperature=temperature,
    )

    results.to_csv(results_csv, index=False)
    results_t.to_csv(results_t_csv, index=False)

    filtered_csv = os.path.join(step_path, "results_t_filtered.csv")
    results_t_filtered = filter_confidence(
        data_path=results_t_csv,
        output_path=filtered_csv,
        threshold=confidence_threshold,
    )

    metrics_path = os.path.join(step_path, "metrics.csv")
    calculate_metrics(filtered_csv, metrics_path)
    df_metrics = pd.read_csv(metrics_path)

    candidate_list_path = os.path.join(step_path, "candidate_list.csv")
    qualified_species = df_metrics[
        (df_metrics["Recall"] >= Recall_Precision) &
        (df_metrics["Precision"] >= Recall_Precision)
    ]["Species"].astype(str).tolist()

    pd.DataFrame({"Species": sorted(qualified_species)}).to_csv(candidate_list_path, index=False)

    gate_path = os.path.join(step_path, "gate")
    os.makedirs(gate_path, exist_ok=True)

    df_allow = gate_from_candidates(
        results_t=results_t,
        results_t_filtered=results_t_filtered,
        candidate_species=qualified_species,
        output_dir=gate_path,
        coverage_threshold=coverage_threshold,
    )

    print(f"\n{'='*70}\n✅ GATE FINAL SPECIES (n={len(df_allow)}):\n  " + (", ".join(df_allow['Species'].astype(str)) or "NONE") + f"\n{'='*70}")

    # =====integrated_f1_scores.csv =====
    integrated_path = "./integrated_f1_scores.csv"
    df_allow.to_csv(integrated_path, index=False)


if __name__ == "__main__":
    main()
