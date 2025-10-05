import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_tensorboard_scalars(event_file, tags):
    ea = EventAccumulator(event_file)
    ea.Reload()
    scalars = {}
    for tag in tags:
        if tag in ea.Tags()["scalars"]:
            scalars[tag] = [s.value for s in ea.Scalars(tag)]
    return scalars


def plot_metrics(all_metrics, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for metric in all_metrics[0].keys():
        plt.figure()
        for fold_idx, metrics in enumerate(all_metrics, 1):
            plt.plot(metrics[metric], label=f"Fold {fold_idx}")
        plt.title(metric)
        plt.xlabel("Epoch")
        plt.ylabel(metric)
        plt.legend()
        plt.savefig(os.path.join(output_dir, f"{metric.replace('/','_')}.png"))
        plt.close()


def aggregate_csv_metrics(input_dir, output_dir, folds=5):
    csv_dir = os.path.join(input_dir, "checkpoints", f"fold{folds}")
    csv_file = [
        f for f in os.listdir(csv_dir) if f.startswith("kfold_metrics") and f.endswith(".csv")
    ]
    results = pd.read_csv(os.path.join(csv_dir, csv_file[0]))

    if not results.empty:
        if "test/loss" in results.columns:
            results = results.drop(columns=["test/loss"])

        results.insert(0, "Fold", [f"Fold {i}" for i in range(1, len(results))] + ["Average"])

        results.to_csv(os.path.join(output_dir, "test_metrics_all_folds.csv"), index=False)

        fig, ax = plt.subplots(figsize=(8, 2 + 0.3 * len(results)))
        ax.axis("off")
        ax.axis("tight")
        table = ax.table(
            cellText=results.round(4).values,
            colLabels=results.columns,
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.2)

        for (row, col), cell in table.get_celld().items():
            if row == len(results):
                cell.set_text_props(weight="bold")

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "test_metrics_table.png"), dpi=200)
        plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--metric-tags",
        nargs="+",
        default=[
            "train/loss",
            "train/f1",
            "train/roc_auc",
            "train/auprc",
            "val/loss",
            "val/f1",
            "val/roc_auc",
            "val/auprc",
        ],
    )
    args = parser.parse_args()

    all_metrics = []

    for fold in range(1, args.folds + 1):
        fold_dir = os.path.join(args.input_dir, f"tensorboard/fold{fold}")
        tb_files = [os.path.join(fold_dir, f) for f in os.listdir(fold_dir) if "tfevents" in f]
        if not tb_files:
            continue

        scalars = load_tensorboard_scalars(tb_files[0], args.tags)
        all_metrics.append(scalars)

    if all_metrics:
        plot_metrics(all_metrics, args.output_dir)

    aggregate_csv_metrics(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
