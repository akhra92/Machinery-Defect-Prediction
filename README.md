# Machine Sound Anomaly Detection

Unsupervised detection of faulty machines from their sound. Models are trained on **normal sound only**,
then flag anything that sounds different. The setup follows the DCASE Task 2 benchmark, so results can be compared
with published work.

## Setup

Linux / GPU server (needs an NVIDIA driver with CUDA 12.8+, check with `nvidia-smi`; for older drivers change
the PyTorch index in `pyproject.toml` to `cu126` or `cu124`):
```bash
uv sync
source .venv/bin/activate
```

Windows: if the project folder path is long, PyTorch's CUDA DLLs fail to load ("filename too long").
Put the venv at a short path instead:
```bash
set UV_PROJECT_ENVIRONMENT=C:\venvs\msad
uv sync
```

Using several GPUs: run one machine type per GPU, e.g.
`CUDA_VISIBLE_DEVICES=1 python -m msad.run --machine valve --model ae`.

## Data: DCASE 2024 Task 2 development set

7 machine types (bearing, fan, gearbox, slider, valve, ToyCar, ToyTrain), 10-second clips at 16 kHz.
Per machine: 1000 normal training clips (990 source domain + 10 target domain, i.e. changed operating conditions),
and 200 labelled test clips (normal and anomaly). License: **CC BY-NC-SA 4.0 (non-commercial)**.

```bash
python scripts/download_dcase2024.py --machines fan          # ~320 MB
python scripts/download_dcase2024.py --machines all          # ~2.2 GB
```

## Train and evaluate

```bash
python -m msad.run --machine fan --model gmm knn ae
python -m msad.run --machine all --model ae
python -m msad.run --machine fan --model ae --set ae.epochs=30 ae.bottleneck_dim=16
```
Outputs go to `outputs/<machine>/<model>/` (model, threshold, per-clip scores, training history), and every run
appends a row to `outputs/results.csv`.

Score new recordings with a trained model:
```bash
python -m msad.predict --model-dir outputs/fan/ae path/to/*.wav
```

No data yet? `python scripts/make_synthetic.py` makes a small fake dataset to test the pipeline.

## Models

| Name | Input | Anomaly score |
|---|---|---|
| `gmm` | mean + std of each log-mel band over the clip | negative log-likelihood under a Gaussian mixture |
| `knn` | same clip statistics | distance to the nearest normal training clip |
| `ae`  | 5 stacked log-mel frames (640-dim) | autoencoder reconstruction error (DCASE baseline) |

## Metrics

- **AUC (source / target)**: how well scores separate normal from anomalous clips. Each domain's normal clips are
  compared against all anomalies. Target AUC shows how robust the model is when conditions change.
- **pAUC**: AUC limited to false-positive rate ≤ 0.1, i.e. performance in the low-false-alarm range that matters in a factory.
- **hmean**: harmonic mean of the three above (DCASE ranking metric).
- **precision / recall / F1**: at the decision threshold, which is the 90th percentile of scores on normal training clips
  (`threshold_percentile` in the config). Raise it to get fewer false alarms.

## Layout

```
configs/default.yaml        all settings
scripts/download_dcase2024.py
scripts/make_synthetic.py
src/msad/data.py            find clips, parse labels and domain from DCASE filenames
src/msad/features.py        log-mel features (cached in data/cache), frame stacking, clip stats
src/msad/models.py          gmm / knn / autoencoder detectors
src/msad/metrics.py         AUC, pAUC, hmean, F1
src/msad/run.py             train + evaluate CLI
src/msad/predict.py         score new files
```

## Working together

Project management lives in Notion (Tasks board, Experiments leaderboard, Meetings & Decisions).

1. Pick a task in Notion, set it to *In progress*.
2. Branch from `main`: `git switch -c <ID>-short-name` (e.g. `MSAD-6-parallel-features`).
3. Commit, push, open a pull request. Fill in the PR template, paste the PR link into the task's *Branch / PR* field and
   set the task to *In review*.
4. The other person reviews; merge with squash, then set the task to *Done*.
5. Log training runs worth keeping in the Notion *Experiments* database (metrics from `outputs/results.csv`).

Never commit `data/` or `outputs/` (already in `.gitignore`); datasets are re-downloaded with the script.

## Next steps

1. Stronger model: a self-supervised classifier (MobileFaceNet/ResNet + ArcFace) trained to predict machine
   attributes, scored by embedding distance. This is the approach most top DCASE systems use.
2. Pretrained audio embeddings (BEATs, PANNs, EAT) + kNN, which helps when there is little data.
3. Your own machines: record normal operation, put the clips in `<data_root>/<machine>/train/`, train, and set the threshold.
