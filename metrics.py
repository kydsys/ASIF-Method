import pandas as pd

def calculate_metrics(df):
    species = df["Actual"].unique()
    # print(species)
    results = []
    correct_predictions = (df["Actual"] == df["Predicted"]).sum()
    total_samples = df.shape[0]
    total_accuracy = correct_predictions / total_samples

    for specie in species:
        tp = len(df[(df["Actual"] == specie) & (df["Predicted"] == specie)])
        fp = len(df[(df["Actual"] != specie) & (df["Predicted"] == specie)])
        fn = len(df[(df["Actual"] == specie) & (df["Predicted"] != specie)])
        tn = len(df[(df["Actual"] != specie) & (df["Predicted"] != specie)])

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

        results.append({
            "Species": specie,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
            "Recall": round(recall, 4) * 100,
            "Precision": round(precision, 4) * 100,
            "F1-Score": round(f1, 4) * 100,
        })

    return results, total_accuracy


def process_results_and_calculate_metrics(data_path, metrics_path):
    df = pd.read_csv(data_path, dtype=str)

    if df.shape[0] > 0 and str(df.iloc[0, 0]).lower().strip() == "filename":
        df = df.iloc[1:].reset_index(drop=True)
    df = df[df.iloc[:, 0].notna()]
    df = df[~df.iloc[:, 0].astype(str).str.lower().isin(["none"])]

    processed_df = df.iloc[:, [0, 1, 2]]
    processed_df.columns = ["filename", "Predicted", "Actual"]

    results, total_accuracy = calculate_metrics(processed_df)
    print(f"Total Accuracy: {total_accuracy:.4f}")

    results_df = pd.DataFrame(results)
    column_order = [
        "Species", "TP", "FP", "FN", "TN",
        "Recall", "Precision", "F1-Score"
    ]
    results_df = results_df[column_order]
    results_df["Species"] = results_df["Species"].astype(str)
    results_df = results_df.sort_values(by="Species")

    results_df.to_csv(metrics_path, index=False)
    print(f"Metrics saved to {metrics_path}")


def main():
    process_results_and_calculate_metrics(
        data_path="Results.csv",
        metrics_path="metrics.csv"
    )


if __name__ == "__main__":
    main()
