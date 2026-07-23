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

The model is used for 3-class document classification.

---

# Experiments

## Experiment 1: Full Fine-Tuning Baseline

The complete DistilBERT model is fine-tuned on the document classification task.

- Total parameters: 66,955,779
- Trainable parameters: 66,955,779
- Trainable percentage: 100%

### Results

| Metric | Score |
|---|---:|
| Test Accuracy | 100.00% |
| Macro-F1 | 1.000 |

The baseline model correctly classified all 371 test documents.

---

## Experiment 2: LoRA Fine-Tuning

LoRA (Low-Rank Adaptation) is used to fine-tune the model while freezing the majority of the pretrained parameters.

- Total parameters: 67,696,134
- Trainable parameters: 740,355
- Trainable percentage: 1.0936%

### Results

| Metric | Score |
|---|---:|
| Test Accuracy | 100.00% |
| Macro-F1 | 1.000 |

The LoRA model correctly classified all 371 test documents.

---

# Full Fine-Tuning vs LoRA

| Metric | Full Fine-Tuning | LoRA |
|---|---:|---:|
| Test Accuracy | 100.00% | 100.00% |
| Macro-F1 | 1.000 | 1.000 |
| Trainable Parameters | 66,955,779 | 740,355 |
| Trainable Percentage | 100.00% | 1.0936% |
| Training Time | 4676.25 sec | 3778.88 sec |

### Key Finding

LoRA achieved the same classification performance as full fine-tuning while
training approximately **98.9% fewer parameters**.

This demonstrates the parameter-efficiency advantage of LoRA for this document
classification task.

> Note: Training time is hardware- and environment-dependent. The primary
> efficiency comparison in this experiment is the reduction in trainable
> parameters.

---

## Classification Performance

Both approaches achieved perfect classification performance on the test set.

### LoRA Confusion Matrix

```text
                   Predicted
                 Inv  PO  Ship

Actual Invoice   125   0    0
Actual PO          0 125    0
Actual Ship        0   0  121
```

---

## Project Structure

The project is organized into separate directories for data, experiments,
trained model outputs, and evaluation results.

```text
DocuTune/
│
├── data/
│   ├── raw/
│   │   └── train-00000-of-00001.parquet
│   │
│   └── processed/
│       ├── train.csv
│       ├── validation.csv
│       └── test.csv
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_baseline_training.ipynb
│   └── 03_lora_training.ipynb
│
├── results/
│   ├── baseline_results.json
│   └── baseline_vs_lora.csv
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Technologies
Python
PyTorch
Hugging Face Transformers
Hugging Face PEFT
LoRA
Pandas
NumPy
Scikit-learn
Jupyter Notebook





