"""Evaluate all Teachable Machine models on the fixed test set.

Extended from the course example `03_evaluate_ai_model.py`:
  * evaluates every model/run folder (model/M0/run1 ... model/M5/run3)
    instead of one hard-coded model path
  * converts OpenCV's BGR images to RGB, because Teachable Machine is trained on RGB
  * reports precision, recall, F1 and the confusion matrix, not only accuracy
  * saves every prediction and all metrics as CSV files for later analysis

Usage (run from the repository root):
    python scripts/evaluate.py                    # evaluate all models
    python scripts/evaluate.py --models M0 M2     # evaluate selected models only
    python scripts/evaluate.py --save-images      # also save correct/wrong images (as in the course code)
"""

import argparse
import csv
import os

import cv2  # Image operations.
import numpy as np  # Array operations.
from tensorflow.keras.models import load_model  # Load a saved Keras model.

CLASSES = ["human", "ai"]  # Must match the test folder names and labels.txt.
IMG_SIZE = 224             # Input size of the Teachable Machine model.
IMAGE_EXT = (".jpg", ".jpeg", ".png")

# Short description of each experiment, written to the summary file.
CONDITIONS = {
    "M0": "baseline (200/200, clean)",
    "M1": "imbalance 1:5 (40 human / 200 ai)",
    "M2": "imbalance 1:10 (20 human / 200 ai)",
    "M3": "label noise 10%",
    "M4": "label noise 20%",
    "M5": "label noise 40%",
}


def load_labels(path):
    """Read labels.txt; remove each label's index and whitespace (as in the course code)."""
    with open(path, "r") as f:
        return [line[2:].strip() for line in f if line.strip()]


def preprocess(image_bgr):
    """Prepare one image exactly as Teachable Machine expects."""
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)  # OpenCV reads BGR; the model expects RGB.
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    image = np.asarray(image, dtype=np.float32)
    return (image / 127.5) - 1  # Scale pixels to [-1, 1].


def style_from_filename(filename, label):
    """Get the art style from a file name such as 'human_baroque_007.jpg'."""
    stem = os.path.splitext(filename)[0]
    return stem[len(label) + 1:stem.rfind("_")]


def load_test_set(test_dir):
    """Load all test images once. The folder name is the ground-truth label."""
    images, records = [], []
    for true_label in CLASSES:
        folder = os.path.join(test_dir, true_label)
        for filename in sorted(os.listdir(folder)):
            if not filename.lower().endswith(IMAGE_EXT):
                continue
            path = os.path.join(folder, filename)
            image_ = cv2.imread(path)  # Original BGR image.
            images.append(preprocess(image_))
            records.append({
                "file": filename,
                "path": path,
                "style": style_from_filename(filename, true_label),
                "true_label": true_label,
            })
    return np.stack(images), records


def compute_metrics(true_labels, pred_labels):
    """Accuracy, per-class precision/recall/F1, macro F1 and the confusion matrix."""
    true_labels, pred_labels = np.array(true_labels), np.array(pred_labels)
    metrics = {"accuracy": float(np.mean(true_labels == pred_labels))}
    f1_scores = []
    for c in CLASSES:
        tp = np.sum((pred_labels == c) & (true_labels == c))
        fp = np.sum((pred_labels == c) & (true_labels != c))
        fn = np.sum((pred_labels != c) & (true_labels == c))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        metrics[f"precision_{c}"] = float(precision)
        metrics[f"recall_{c}"] = float(recall)
        metrics[f"f1_{c}"] = float(f1)
        f1_scores.append(f1)
    metrics["macro_f1"] = float(np.mean(f1_scores))
    # Confusion matrix entries: cm_<true>_<pred>.
    for t in CLASSES:
        for p in CLASSES:
            metrics[f"cm_{t}_{p}"] = int(np.sum((true_labels == t) & (pred_labels == p)))
    return metrics


def save_images(records, pred_labels, confidences, out_dir):
    """Save each test image with its prediction into correct/ or wrong/ (course code behaviour)."""
    for rec, pred, conf in zip(records, pred_labels, confidences):
        folder = "correct" if pred == rec["true_label"] else "wrong"
        os.makedirs(os.path.join(out_dir, folder), exist_ok=True)
        image_ = cv2.imread(rec["path"])
        image_ = cv2.putText(image_, pred, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        image_ = cv2.putText(image_, f"{conf * 100:.0f}%", (150, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.imwrite(os.path.join(out_dir, folder, rec["file"]), image_)


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--test-dir", default="data/test", help="Folder with human/ and ai/ test images")
    parser.add_argument("--model-dir", default="model", help="Folder with M*/run*/keras_model.h5")
    parser.add_argument("--out", default="results", help="Output folder for CSV files")
    parser.add_argument("--models", nargs="*", help="Evaluate only these models, e.g. M0 M2")
    parser.add_argument("--save-images", action="store_true", help="Save correct/wrong images per model run")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    images, records = load_test_set(args.test_dir)
    print(f"Test set: {len(records)} images "
          f"({sum(r['true_label'] == 'human' for r in records)} human, "
          f"{sum(r['true_label'] == 'ai' for r in records)} ai)")

    model_names = args.models or sorted(os.listdir(args.model_dir))
    prediction_rows, metric_rows = [], []

    for model_name in model_names:
        for run in sorted(os.listdir(os.path.join(args.model_dir, model_name))):
            run_dir = os.path.join(args.model_dir, model_name, run)
            model = load_model(os.path.join(run_dir, "keras_model.h5"), compile=False)
            class_names = load_labels(os.path.join(run_dir, "labels.txt"))

            # Predict class scores for all test images; select the highest-scoring class.
            scores = model.predict(images, batch_size=32, verbose=0)
            pred_index = np.argmax(scores, axis=1)
            pred_labels = [class_names[i] for i in pred_index]
            confidences = scores[np.arange(len(scores)), pred_index]

            for rec, pred, score in zip(records, pred_labels, scores):
                row = {"model": model_name, "run": run, "file": rec["file"], "style": rec["style"],
                       "true_label": rec["true_label"], "pred_label": pred}
                for name, s in zip(class_names, score):
                    row[f"score_{name}"] = round(float(s), 6)
                prediction_rows.append(row)

            metrics = compute_metrics([r["true_label"] for r in records], pred_labels)
            metric_rows.append({"model": model_name, "run": run, **metrics})
            print(f"{model_name}/{run}: accuracy {metrics['accuracy']:.3f}, "
                  f"recall human {metrics['recall_human']:.3f}, recall ai {metrics['recall_ai']:.3f}, "
                  f"macro F1 {metrics['macro_f1']:.3f}")

            if args.save_images:
                save_images(records, pred_labels, confidences, os.path.join(args.out, "images", model_name, run))

    # Mean and standard deviation over the runs of each model.
    summary_rows = []
    keys = ["accuracy", "recall_human", "recall_ai", "precision_human", "precision_ai", "macro_f1"]
    for model_name in model_names:
        runs = [m for m in metric_rows if m["model"] == model_name]
        row = {"model": model_name, "condition": CONDITIONS.get(model_name, ""), "n_runs": len(runs)}
        for k in keys:
            values = np.array([m[k] for m in runs])
            row[f"{k}_mean"] = round(float(values.mean()), 4)
            row[f"{k}_std"] = round(float(values.std(ddof=1)) if len(values) > 1 else 0.0, 4)
        summary_rows.append(row)

    write_csv(os.path.join(args.out, "predictions.csv"), prediction_rows)
    write_csv(os.path.join(args.out, "metrics_per_run.csv"), metric_rows)
    write_csv(os.path.join(args.out, "metrics_summary.csv"), summary_rows)

    print("\n===== Summary (mean ± std over runs) =====")
    print(f"{'model':<6}{'condition':<38}{'accuracy':>16}{'recall human':>16}{'recall ai':>16}{'macro F1':>16}")
    for r in summary_rows:
        cells = [f"{r[k + '_mean']:.3f} ± {r[k + '_std']:.3f}" for k in ["accuracy", "recall_human", "recall_ai", "macro_f1"]]
        print(f"{r['model']:<6}{r['condition']:<38}" + "".join(f"{c:>16}" for c in cells))
    print(f"\nSaved CSV files to '{args.out}/'.")


if __name__ == "__main__":
    main()
