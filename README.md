# AI-art-detection

Can a small image classifier tell a human painting from an AI-generated one, and what happens to it when the training data is not clean?

Project for EF2039 AI Programming (PA04). A human vs. AI artwork classifier is trained with Teachable Machine under two kinds of data problems:

1. **Class imbalance**: what if there are far fewer human paintings than AI images in the training set?
2. **Label noise**: what if some training images have the wrong label?

Each training set was trained 3 times, and all 18 models were tested on the same 1,000 test images.

## Results

Mean ± std over the 3 runs (test set: 500 human, 500 AI).

| Model | Training set | Accuracy | Recall (human) | Recall (AI) | Macro F1 |
|---|---|---|---|---|---|
| M0 | 200 human / 200 AI (baseline) | 0.933 ± 0.004 | 0.928 ± 0.004 | 0.937 ± 0.005 | 0.933 ± 0.004 |
| M1 | 40 / 200 (1:5) | 0.792 ± 0.011 | 0.588 ± 0.021 | 0.997 ± 0.001 | 0.783 ± 0.012 |
| M2 | 20 / 200 (1:10) | 0.669 ± 0.012 | 0.338 ± 0.023 | 0.999 ± 0.001 | 0.628 ± 0.017 |
| M3 | 200 / 200, 10% labels flipped | 0.879 ± 0.003 | 0.861 ± 0.026 | 0.897 ± 0.024 | 0.879 ± 0.003 |
| M4 | 200 / 200, 20% labels flipped | 0.815 ± 0.012 | 0.789 ± 0.027 | 0.841 ± 0.003 | 0.815 ± 0.012 |
| M5 | 200 / 200, 40% labels flipped | 0.637 ± 0.037 | 0.608 ± 0.039 | 0.665 ± 0.060 | 0.636 ± 0.037 |

![RQ1 and RQ2 results](results/fig1_research_questions.png)

![Accuracy per art style](results/fig2_style_accuracy.png)

## What's in here

```
AI-art-detection/
├── environment.yml       conda environment
├── scripts/
│   ├── make_splits.py    builds data/ from the original Kaggle zip
│   ├── evaluate.py       tests all 18 models -> results/*.csv
│   └── make_figures.py   draws the two figures above
├── data/
│   ├── train/M0 ~ M5/    images uploaded to Teachable Machine (human/, ai/)
│   ├── test/             1,000 test images, never used for training
│   ├── manifest_train.csv
│   └── manifest_test.csv
├── model/M0 ~ M5/run1 ~ run3/   keras_model.h5 + labels.txt (18 models)
└── results/              predictions, metrics and figures
```

## How to run

Only conda is needed. Everything runs on CPU (tested on Windows and Linux with the same numbers).

Hardware requirements: no GPU, 8 GB RAM recommended (`evaluate.py` peaks at about 4 GB), about 2 GB of disk space (1.5 GB for the conda environment, 120 MB for this repository). Evaluating all 18 models takes under 3 minutes on a 2-core CPU.

```bash
git clone https://github.com/dongy2on/AI-art-detection.git
cd AI-art-detection

conda env create -f environment.yml
conda activate ai_art

python scripts/evaluate.py      # prints the results table above, saves CSVs to results/
python scripts/make_figures.py  # saves the two figures to results/
```

`evaluate.py` also takes `--models M0 M2` (only some models) and `--save-images` (saves correct/wrong images like the course example code).

## Data

Dataset: [AI-ArtBench](https://www.kaggle.com/datasets/ravidussilva/real-ai-art) from Kaggle (MIT License). Its human paintings come from ArtBench-10, so check ArtBench-10's terms too if you reuse the images.

- **human**: human paintings, 256×256
- **ai**: images made with Latent Diffusion, 256×256

The dataset also has Stable Diffusion images, but they are 768×768. They are left out, otherwise the model could just learn "bigger image = AI".

Images are sampled equally from all 10 styles (art nouveau, baroque, expressionism, impressionism, post-impressionism, realism, renaissance, romanticism, surrealism, ukiyo-e) with a fixed seed (2039):

- train: 20 images per style per class from the official **train** split -> 200 human + 200 AI
- test: 50 images per style per class from the official **test** split -> 500 human + 500 AI

Since train and test come from different official splits, no test image was seen during training.

The 6 training sets only differ in the factor being tested. The smaller sets are subsets of the bigger ones, and the images flipped in M3 are also flipped in M4 and M5.

| Model | human/ folder | ai/ folder |
|---|---|---|
| M0 | 200 | 200 |
| M1 | 40 (4 per style) | 200 |
| M2 | 20 (2 per style) | 200 |
| M3 | 180 human + 20 AI (wrong) | 180 AI + 20 human (wrong) |
| M4 | 160 + 40 wrong | 160 + 40 wrong |
| M5 | 120 + 80 wrong | 120 + 80 wrong |

File names start with the real class, so `ai_baroque_008.jpg` inside `train/M5/human/` is an AI image with a wrong label. The two manifest CSVs list every image with its original path in the Kaggle zip.

To rebuild `data/` yourself, download the Kaggle zip (~11 GB) and run:

```bash
python scripts/make_splits.py --zip path/to/real-ai-art.zip --out data_rebuilt
```

## Training

All models were made with [Teachable Machine](https://teachablemachine.withgoogle.com/) (Image Project, Standard image model). It fine-tunes only the last layers of a pretrained MobileNet.

- Classes: `human` (class 1), `ai` (class 2)
- Default settings for every model: 50 epochs, batch size 16, learning rate 0.001
- Teachable Machine keeps 15% of the uploaded images aside for its own validation
- Each training set was trained 3 times (run1 ~ run3), exported as Keras `.h5`

If you retrain, upload `data/train/<model>/human` and `data/train/<model>/ai` with the same settings. Your numbers will be close but not identical because training is random.

## Evaluation

`evaluate.py` is based on the course example `03_evaluate_ai_model.py`, with a few changes:

- it loops over all 18 models instead of one hard-coded path
- it converts images from BGR (OpenCV) to RGB, because Teachable Machine trains on RGB
- it reports precision, recall, F1 and the confusion matrix, not just accuracy
- it saves everything to CSV

Images are resized to 224×224 and scaled to [-1, 1], same as Teachable Machine's own example code.
