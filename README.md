# DocuTune

## Parameter-Efficient Text Classification with LoRA

DocuTune is an experimental machine learning project that investigates whether
parameter-efficient fine-tuning using LoRA can achieve comparable text
classification performance to full fine-tuning while updating significantly
fewer model parameters.

---

## Objective

This project compares two fine-tuning strategies for document classification:

1. Full Fine-Tuning
2. LoRA (Low-Rank Adaptation)

The primary research question is:

> Can LoRA achieve comparable classification performance while training
> significantly fewer parameters than full fine-tuning?

---

## Dataset

The project uses the Company Documents dataset containing 2,469 documents
across three categories:

- Invoices
- Purchase Orders
- Shipping Orders

### Dataset Split

| Split | Samples |
|---|---:|
| Training | 1,728 |
| Validation | 370 |
| Testing | 371 |

---

## Model

### Base Model

`distilbert-base-uncased`

### Baseline

Full fine-tuning of DistilBERT for 3-class document classification.

- Total parameters: 66,955,779
- Trainable parameters: 66,955,779
- Trainable percentage: 100%

---

## Baseline Results

| Metric | Score |
|---|---:|
| Test Accuracy | 100.00% |
| Macro-F1 | 1.000 |

### Classification Results

| Class | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| Invoices | 1.00 | 1.00 | 1.00 |
| Purchase Orders | 1.00 | 1.00 | 1.00 |
| Shipping Orders | 1.00 | 1.00 | 1.00 |

The baseline model correctly classified all 371 test documents.

---

## Project Structure

```text
DocuTune/
│
├── data/
│   ├── raw/
│   └── processed/
│       ├── train.csv
│       ├── validation.csv
│       └── test.csv
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   └── 02_baseline_training.ipynb
│
├── results/
│   └── baseline_results.json
│
├── requirements.txt
├── .gitignore
└── README.md