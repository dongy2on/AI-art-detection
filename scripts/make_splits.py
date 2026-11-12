"""Build the train/test subsets for all experiments from the AI-ArtBench zip.

The script reads images directly from the downloaded zip (no full extraction),
samples a fixed number of images per art style with a fixed seed, and writes
one folder per experiment so that each folder can be uploaded to
Teachable Machine as-is.

Classes:
    human : human-made artworks (ArtBench-10 part of AI-ArtBench)
    ai    : AI-generated artworks (Latent Diffusion part, 256x256)

Experiments (train sets; the test set is shared by all models):
    M0 : 200 human / 200 ai, clean labels              (baseline)
    M1 :  40 human / 200 ai, clean labels              (RQ1, imbalance 1:5)
    M2 :  20 human / 200 ai, clean labels              (RQ1, imbalance 1:10)
    M3 : 200 human / 200 ai, 10% labels flipped        (RQ2, label noise)
    M4 : 200 human / 200 ai, 20% labels flipped        (RQ2, label noise)
    M5 : 200 human / 200 ai, 40% labels flipped        (RQ2, label noise)

Usage:
    python scripts/make_splits.py --zip /tmp/artbench/real-ai-art.zip --out data
"""

import argparse
import csv
import os
import random
import shutil
import zipfile
from collections import Counter, defaultdict

ROOT = "Real_AI_SD_LD_Dataset"
STYLES = [
    "art_nouveau", "baroque", "expressionism", "impressionism",
    "post_impressionism", "realism", "renaissance", "romanticism",
    "surrealism", "ukiyo_e",
]
LABELS = ["human", "ai"]
IMAGE_EXT = (".jpg", ".jpeg", ".png")

SEED = 2039
N_TRAIN_PER_STYLE = 20     # per class -> 200 images per class
N_TEST_PER_STYLE = 50      # per class -> 500 images per class
IMBALANCE = {"M1": 4, "M2": 2}            # human images kept per style
NOISE = {"M3": 0.10, "M4": 0.20, "M5": 0.40}  # fraction of flipped labels per class


def source_folder(split, label, style):
    """Return the folder path inside the zip for a split/class/style."""
    if label == "human":
        return f"{ROOT}/{split}/{style}"
    # AI folders use 'ukiyo-e' while human folders use 'ukiyo_e'.
    ai_style = "ukiyo-e" if style == "ukiyo_e" else style
    return f"{ROOT}/{split}/AI_LD_{ai_style}"


def index_zip(zf):
    """Map each folder in the zip to a sorted list of its image paths."""
    folders = defaultdict(list)
    for name in zf.namelist():
        if name.lower().endswith(IMAGE_EXT):
            folders[os.path.dirname(name)].append(name)
    for paths in folders.values():
        paths.sort()  # sorting makes sampling independent of zip order
    return folders


def sample(folders, rng, split, n_per_style):
    """Sample n_per_style images per style and class. Returns a list of dicts."""
    records = []
    for label in LABELS:
        for style in STYLES:
            folder = source_folder(split, label, style)
            pool = folders.get(folder, [])
            if len(pool) < n_per_style:
                raise SystemExit(f"Not enough images in {folder}: {len(pool)}")
            for k, path in enumerate(rng.sample(pool, n_per_style)):
                ext = os.path.splitext(path)[1].lower()
                records.append({
                    "true_label": label,
                    "style": style,
                    "source": path,  # original path inside the zip
                    # Short name: some original names are too long for Windows paths.
                    "file": f"{label}_{style}_{k:03d}{ext}",
                })
    return records


def write_image(zf, record, out_dir, label_folder):
    """Copy one image from the zip into out_dir/label_folder/."""
    dst_dir = os.path.join(out_dir, label_folder)
    os.makedirs(dst_dir, exist_ok=True)
    with zf.open(record["source"]) as src, open(os.path.join(dst_dir, record["file"]), "wb") as dst:
        shutil.copyfileobj(src, dst)


def build_train_sets(base):
    """Create the train-label assignment for every experiment.

    Subsets are nested so that experiments differ only in the studied factor:
    M2's human images are a subset of M1's, and the images flipped in M3 are
    also flipped in M4 and M5.
    """
    rng = random.Random(SEED + 1)
    experiments = {}

    # M0: baseline, clean labels.
    experiments["M0"] = [dict(r, train_label=r["true_label"], flipped=0) for r in base]

    # M1, M2: keep all ai images, keep the first k human images per style.
    for name, k in IMBALANCE.items():
        kept, seen = [], Counter()
        for r in base:
            if r["true_label"] == "human":
                if seen[r["style"]] >= k:
                    continue
                seen[r["style"]] += 1
            kept.append(dict(r, train_label=r["true_label"], flipped=0))
        experiments[name] = kept

    # M3-M5: flip the same random order of images in each class (nested).
    flip_order = {}
    for label in LABELS:
        idx = [i for i, r in enumerate(base) if r["true_label"] == label]
        rng.shuffle(idx)
        flip_order[label] = idx
    for name, p in NOISE.items():
        flipped = set()
        for label in LABELS:
            n_flip = round(p * len(flip_order[label]))
            flipped.update(flip_order[label][:n_flip])
        rows = []
        for i, r in enumerate(base):
            if i in flipped:
                other = "ai" if r["true_label"] == "human" else "human"
                rows.append(dict(r, train_label=other, flipped=1))
            else:
                rows.append(dict(r, train_label=r["true_label"], flipped=0))
        experiments[name] = rows

    return experiments


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--zip", required=True, help="Path to real-ai-art.zip downloaded from Kaggle")
    parser.add_argument("--out", default="data", help="Output folder (default: data)")
    parser.add_argument("--archive", action="store_true", help="Also create <out>.zip for easy download")
    args = parser.parse_args()

    if os.path.exists(args.out):
        raise SystemExit(f"Output folder '{args.out}' already exists. Remove it first to rebuild.")

    zf = zipfile.ZipFile(args.zip)
    folders = index_zip(zf)
    rng = random.Random(SEED)

    # Train images are sampled from the official train split, test from the official test split.
    train_base = sample(folders, rng, "train", N_TRAIN_PER_STYLE)
    test_set = sample(folders, rng, "test", N_TEST_PER_STYLE)
    experiments = build_train_sets(train_base)

    fields = ["experiment", "file", "train_label", "true_label", "flipped", "style", "source"]
    with open(os.path.join(os.makedirs(args.out) or args.out, "manifest_train.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for name, rows in experiments.items():
            for r in rows:
                write_image(zf, r, os.path.join(args.out, "train", name), r["train_label"])
                writer.writerow(dict(r, experiment=name))

    with open(os.path.join(args.out, "manifest_test.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "true_label", "style", "source"])
        writer.writeheader()
        for r in test_set:
            write_image(zf, r, os.path.join(args.out, "test"), r["true_label"])
            writer.writerow({k: r[k] for k in ["file", "true_label", "style", "source"]})

    # Summary: number of images in each train-label folder.
    print(f"{'set':<10}{'human':>8}{'ai':>8}{'flipped':>9}")
    for name, rows in experiments.items():
        c = Counter(r["train_label"] for r in rows)
        print(f"{'train/' + name:<10}{c['human']:>8}{c['ai']:>8}{sum(r['flipped'] for r in rows):>9}")
    c = Counter(r["true_label"] for r in test_set)
    print(f"{'test':<10}{c['human']:>8}{c['ai']:>8}{0:>9}")

    if args.archive:
        path = shutil.make_archive(args.out, "zip", root_dir=".", base_dir=args.out)
        print(f"Archive written: {path} ({os.path.getsize(path) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
