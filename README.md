# DiffAlpha

## File Structure

```text
DiffAlpha_release/
├── README.md
├── requirements.txt
├── .gitignore
├── utils/
│   ├── __init__.py
│   ├── model.py
│   ├── train.py
│   ├── infer.py
│   └── augmentation.py
├── data/                     # Initially empty
└── output/                   # Initially empty
```

| File | Purpose |
| --- | --- |
| `utils/model.py` | Model script |
| `utils/train.py` | Training script |
| `utils/infer.py` | Inference script |
| `utils/augmentation.py` |  Augmentation script
|
 |
|  |
| `requirements.txt` | Runtime dependencies; testing additionally requires `pytest`. |


## Installation

Python 3.12 is recommended; Python 3.10 and later are supported.


### 1. Data Source

Use daily Chinese A-share market data and historical CSI 300 constituent data from the [official BaoStock Python API](https://baostock.com/mainContent?file=pythonAPI.md):

| API | Data |
| --- | --- |
| `query_trade_dates` | Shared trading calendar. |
| `query_hs300_stocks(date=...)` | CSI 300 constituent snapshot for a specified historical date. |
| `query_history_k_data_plus` | Daily OHLC, trading volume, turnover amount, adjustment flag, trading status, and ST flag. |

Price requests use `frequency="d"` and backward adjustment with `adjustflag="1"`.


The raw cache must contain `manifest.json` and the trading calendar, historical constituent, and per-stock price CSV files referenced in it.


| File | Anchor/Label Period | Purpose |
| --- | --- | --- |
| `data/processed/teacher.npz` | 2019–2021 | Generator training and training-period preprocessing. |
| `data/processed/diagnostic.npz` | 2022 | Generator diagnostics; must not be used to fit teacher preprocessing. |
| `data/processed/student.npz` | 2023 | Generate candidate pools and construct downstream augmentation targets. |
| `data/processed/eval.npz` | 2024 |  |

### 

## Training and Inference

### 1. Train the Teacher Generator

```bash
python -m utils.train \
  --data data/processed/teacher.npz \
  --output output/teacher \
  --steps 4800 --batch 8 --width 16 --timesteps 40 --seed 9811
```

### 2. Generate the Raw Candidate Pool

```bash
python -m utils.infer \
  --checkpoint output/teacher/generator.pt \
  --data data/processed/student.npz \
  --output output/student_paths.npz \
  --draws 64 --sample-batch 2 --seed 9811
```
## Augmentation Strategy

```bash
python -m utils.augmentation \
  --data data/processed/student.npz \
  --paths output/student_paths.npz \
  --output output/student_augmented.npz \
  --amplitude 2 --synthetic-weight 0.1
```