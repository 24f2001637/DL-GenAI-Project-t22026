# Smart MCQ Solver & Retrieval-Augmented Generation (RAG) System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/Transformers-DeBERTa--v3-FFD21E.svg?logo=huggingface&logoColor=black)](https://huggingface.co/microsoft/deberta-v3-base)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Course](https://img.shields.io/badge/IITM%20BS-Deep%20Learning%20%26%20GenAI-003366.svg)](https://study.iitm.ac.in/ds/)
[![Evaluation Metric](https://img.shields.io/badge/Metric-MAP%403-success.svg)](#evaluation-metric)
[![Best Kaggle Score](https://img.shields.io/badge/Kaggle%20LB-0.75976%20MAP%403-brightgreen.svg)](#comparative-performance--engineering-trade-offs)

> **IIT Madras BS Degree in Data Science and Applications**  
> **Course:** Deep Learning & Generative AI (Term T2 2026)  
> **Student Roll Number:** `24f2001637`  
> **Repository:** [`24f2001637/DL-GenAI-Project-t22026`](https://github.com/24f2001637/DL-GenAI-Project-t22026)

---

## Executive Summary

The **Smart MCQ Answering System** is an end-to-end deep learning and retrieval-augmented generation (RAG) system engineered to evaluate and rank complex multiple-choice science and physics questions. Given a question prompt and five candidate choices (**A, B, C, D, E**), the system predicts and ranks the top three most probable answers evaluated via **Mean Average Precision at Rank 3 (MAP@3)**.

To rigorously address this challenge, the project implemented and benchmarked **three fundamentally distinct technical techniques across three standalone notebooks**:
1. **Classical ML & Recurrent Deep Learning from Scratch** (TF-IDF baselines + 2.9M-parameter Bi-LSTM with Self-Attention).
2. **Pre-Trained Discriminative Transformer Fine-Tuning** (`microsoft/deberta-v3-base` 5-way multiple-choice head).
3. **Advanced Retrieval-Augmented Generation (RAG) with Quantized LLM** (FAISS dense vector search + Cross-Encoder reranker + 4-bit `Meta-Llama-3-8B-Instruct`).

The best-performing model—**fine-tuned DeBERTa-v3**—achieved a **Kaggle Public Leaderboard MAP@3 score of 0.75976**, and is integrated into a multi-engine, interactive **Streamlit web application**.

---

## Key Highlights & System Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                        Input Question & Options (A-E)                  │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    │                             │
                    ▼                             ▼
   ┌─────────────────────────────────┐   ┌───────────────────────────────┐
   │ Dynamic RAG Retrieval Pipeline  │   │ Direct Transformer Encoder    │
   │  - Pre-Scraped Local Corpus     │   │  - Paired (Prompt, Option_i)  │
   │  - Live MediaWiki Search API    │   │  - Token Length Budgeting     │
   │  - Dense FAISS / Cross-Encoder  │   └──────────────┬────────────────┘
   └────────────────┬────────────────┘                  │
                    │ (Augmented Context)               │
                    └──────────────┬────────────────────┘
                                   ▼
          ┌──────────────────────────────────────────────────┐
          │  Multi-Model Inference & Scoring Engine          │
          │   1. Fine-Tuned DeBERTa-v3 Multiple-Choice       │
          │   2. SentenceTransformers (all-MiniLM-L6-v2)     │
          │   3. Zero-Shot BART-Large-MNLI NLI Pipeline      │
          │   4. TF-IDF Cosine Similarity Baseline           │
          └────────────────────────┬─────────────────────────┘
                                   ▼
          ┌──────────────────────────────────────────────────┐
          │  Softmax Probability Distribution & Ranking      │
          │   - Choice Probabilities (A - E)                 │
          │   - Top-3 Ranked Candidates (e.g., "A D E")      │
          │   - MAP@3 Formatted Output (`submission.csv`)    │
          └──────────────────────────────────────────────────┘
```

- **Three Independent Technical Paradigms:** Implemented and compared across dedicated notebooks (V1: Classical/Bi-LSTM, V2: DeBERTa-v3, V3: Dense RAG with LLaMA-3).
- **Fine-Tuned DeBERTa-v3 Multiple-Choice Head:** Pairs question prompts with each candidate answer `(prompt, option_i)`, encodes them through `microsoft/deberta-v3-base`, and maps pooled representations to scalar logits for softmax ranking.
- **Dynamic Retrieval-Augmented Generation (RAG):** Context is pulled dynamically from a curated local corpus of 180+ pre-scraped Wikipedia science articles, backed by a live fallback to the MediaWiki REST API.
- **Multi-Engine Inference Support:** Runtime switching between DeBERTa-v3, dense sentence transformers (`all-MiniLM-L6-v2`), zero-shot BART-MNLI, and TF-IDF vectorizers.
- **Production Streamlit Application:** Includes single-question inference with confidence charts, bulk CSV test-set processing with progress tracking, model training loss/accuracy dashboards, and searchable dataset explorers.

---

## Evaluation Metric

The competition and evaluation metric is **Mean Average Precision at Rank 3 (MAP@3)**:

$$\text{MAP@3} = \frac{1}{U} \sum_{u=1}^{U} \sum_{k=1}^{\min(n, 3)} P(k) \times \text{rel}(k)$$

Where:
- $U$ is the total number of evaluated questions ($500$ in the test set).
- $P(k)$ is the precision at cut-off $k$.
- $\text{rel}(k)$ is an indicator function equal to $1$ if the item at rank $k$ is the ground-truth correct option, and $0$ otherwise.
- For each question, up to $3$ distinct predictions are submitted formatted as space-delimited letters (e.g., `A D E`).

### Scoring Example

Assuming the ground-truth answer is **A**:

| Submitted Prediction | AP@3 Score | Explanation |
| :---: | :---: | :--- |
| `A B C` | **1.000** | Correct answer at Rank 1 ($1/1$) |
| `B A C` | **0.500** | Correct answer at Rank 2 ($1/2$) |
| `C D A` | **0.333** | Correct answer at Rank 3 ($1/3$) |
| `B C D` | **0.000** | Correct answer not present in top 3 |

---

## Three Technical Techniques & Notebook Implementations

The project investigated three independent modeling paradigms, each packaged into a distinct, reproducible notebook:

### 1. Technique 1: Classical Machine Learning & Custom Deep Learning from Scratch
- **Notebook:** [`notebooks/dl-24f2001637-notebook-t22026 - V1.ipynb`](notebooks/dl-24f2001637-notebook-t22026%20-%20V1.ipynb)
- **Methodology & Components:**
  - **TF-IDF + K-Nearest Neighbors (KNN):** Vectorizes combined prompt-option text with $n$-gram range (1, 2) up to 50,000 features. Evaluates $k=15$ neighbors via cosine distance to retrieve training instance labels.
  - **TF-IDF + Logistic Regression:** Multi-class linear classifier trained directly on TF-IDF vectors with $L_2$ regularization ($C=1.0$).
  - **Custom Bi-LSTM with Multi-Head Self-Attention (2.9M parameters):**
    - **Architecture:** Word Embedding layer ($V \to 128\text{-d}$) $\to$ 2-layer Bidirectional LSTM (256 hidden units/direction, $512\text{-d}$ output) $\to$ Self-Attention module (linear projection $512 \to 1$ with softmax weighting) $\to$ Scorer MLP ($512 \to 128 \to 1$) with LayerNorm, ReLU, and Dropout ($p=0.3$).
    - **Training:** Trained from scratch on the dataset vocabulary for 25 epochs using AdamW ($\text{lr}=3\times 10^{-4}$, weight decay $10^{-4}$) and gradient clipping at 1.0.
    - **Convergence:** Gradually learned language representations, climbing from $\text{MAP@3}=0.57$ at epoch 1 to $0.97$ at epoch 7, reaching $1.00$ at epoch 22.

---

### 2. Technique 2: Pre-Trained Discriminative Transformer Fine-Tuning
- **Notebook:** [`notebooks/dl-24f2001637-notebook-t22026 - V2.ipynb`](notebooks/dl-24f2001637-notebook-t22026%20-%20V2.ipynb) *(and root [`DL-24f2001637-notebook-t22026.ipynb`](DL-24f2001637-notebook-t22026.ipynb))*
- **Methodology & Components:**
  - **Backbone:** `microsoft/deberta-v3-base` (183.8M parameters) with disentangled attention mechanisms.
  - **Multiple-Choice Formulation:** Flattens 5 prompt-option pairs $(P, O_A), \dots, (P, O_E)$ into sequence-pair tokens separated by `[SEP]`. The `[CLS]` token representation is extracted and passed through a linear classification layer to output 5 scalar logits.
  - **Optimization:** Fine-tuned using PyTorch Automatic Mixed Precision (AMP `fp16`), AdamW ($\text{lr}=10^{-5}$), cosine learning rate schedule with 10% warmup, effective batch size 16 (batch $2 \times$ gradient accumulation 8), and gradient clipping at 1.0 over 7 epochs.
  - **Results:** Benefited immensely from prior linguistic pre-training—converged rapidly to **Validation MAP@3 = 1.00 in just 3 epochs**.
  - **Kaggle Leaderboard:** Achieved the highest score among all approaches on the hidden test set with **Public Leaderboard MAP@3 = 0.75976**.

---

### 3. Technique 3: Advanced Two-Stage Dense RAG with 4-Bit Quantized Generative LLM
- **Notebook:** [`notebooks/dl-24f2001637-notebook-t22026 - V3.ipynb`](notebooks/dl-24f2001637-notebook-t22026%20-%20V3.ipynb)
- **Methodology & Components:**
  - **Offline Document Chunking & Indexing:** 190 scientific Wikipedia articles split into 700-character passages with 200-character overlap ($10,535$ chunks) combined with $2,000$ training QA cards = **$12,535$ total knowledge chunks**.
  - **Dense Vector Search:** All passages embedded using `BAAI/bge-base-en-v1.5` ($768\text{-d}$ vectors) and indexed in a FAISS inner product / cosine database.
  - **Neural Cross-Encoder Reranker:** Online query retrieves top 20 passages from FAISS, which are reranked using `cross-encoder/ms-marco-MiniLM-L-12-v2` down to the top 7 most informative passages.
  - **Generative LLM Answering:** Retrieved context, question prompt, and 5 candidate options are formatted into a structured prompt fed to **`meta-llama/Meta-Llama-3-8B-Instruct`**, loaded in 4-bit NormalFloat4 (NF4) quantization via `bitsandbytes`.
  - **Option Parsing:** Regex parsers extract the model's ranked preference over options A–E to output MAP@3 predictions.

---

## Comparative Performance & Engineering Trade-offs

The following table summarizes validation results on the 200-sample split alongside test set generalization and resource trade-offs (based on the official project report [`reports/24f2001637_DG_T22026.pdf`](reports/24f2001637_DG_T22026.pdf)):

| Technique / Model | Notebook | Architectural Paradigm | Parameter Count | Training Regime | Val Acc | Val F1 | Val MAP@3 | Kaggle Test MAP@3 | Inference Speed |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **TF-IDF + KNN** | [V1](notebooks/dl-24f2001637-notebook-t22026%20-%20V1.ipynb) | Instance Retrieval | — | No Training | 1.00 | 1.00 | 1.00 | Baseline | < 5 ms / q |
| **TF-IDF + Logistic Reg.** | [V1](notebooks/dl-24f2001637-notebook-t22026%20-%20V1.ipynb) | Linear Classifier | ~50k weights | Convex Optim. | 1.00 | 1.00 | 1.00 | Baseline | < 5 ms / q |
| **Bi-LSTM + Self-Attention** | [V1](notebooks/dl-24f2001637-notebook-t22026%20-%20V1.ipynb) | Recurrent Deep Net (Scratch) | 2.9M | 25 Epochs | 1.00 | 1.00 | 1.00 | ~0.55 – 0.65 | ~25 ms / q |
| **DeBERTa-v3 Fine-Tuned** | [V2](notebooks/dl-24f2001637-notebook-t22026%20-%20V2.ipynb) | Pre-trained Transformer Head | 183.8M | 7 Epochs (Best: Ep 3) | **1.00** | **1.00** | **1.00** | **0.75976 (Best)** | ~60 ms / q |
| **Dense RAG + LLaMA-3-8B** | [V3](notebooks/dl-24f2001637-notebook-t22026%20-%20V3.ipynb) | Vector Search + GenAI LLM | ~8B (4-bit NF4) | Pre-trained / Zero-Shot | 1.00* | — | 1.00* | ~0.70 – 0.74 | ~3,200 ms / q |

*\* RAG validated on a 10-sample probe due to generative LLM execution time.*

### Key Engineering Insights

1. **The Power of Transfer Learning:** DeBERTa-v3 converged to a perfect validation score in only **3 epochs**, whereas the custom Bi-LSTM model required **22 epochs** to learn lexical semantics from randomly initialized weights.
2. **Memorization vs. Generalization:** TF-IDF baselines achieved perfect validation scores because the 200-sample validation split contained lexical overlap with the training set (KNN could match near-identical prompts). However, on the 500-sample unseen Kaggle test set, DeBERTa-v3's deep contextual understanding significantly outperformed all classical and recurrent approaches (**0.75976 MAP@3**).
3. **Inference Latency Trade-Off:** DeBERTa-v3 discriminative scoring is exceptionally fast (~60 ms per question), whereas RAG generation takes over 3 seconds per question due to dense vector indexing, neural cross-encoder reranking, and autoregressive LLM token decoding.

---

## Milestone Progression & Academic Curriculum

In addition to the three standalone technique notebooks, the project fulfilled all academic milestone deliverables tracking core deep learning competencies:

```
 Milestone 1                  Milestone 2                  Milestone 3                 Final System
┌──────────────────────┐     ┌──────────────────────┐     ┌──────────────────────┐    ┌──────────────────────┐
│ Classical Baselines  │ ──► │ Transformer Probing  │ ──► │ Dense RAG & FAISS    │──► │ Fine-Tuned DeBERTa-v3│
│ - EDA & Cleaning     │     │ - BERT / Attention   │     │ - Vector DB Search   │    │ - Full 5-way MC Head │
│ - TF-IDF Vectorizer  │     │ - Sentence-BERT      │     │ - Cross-Encoder      │    │ - Streamlit App      │
│ - Cosine Similarity  │     │ - Zero-Shot MNLI     │     │ - Adversarial RAG    │    │ - Batch Submission   │
└──────────────────────┘     └──────────────────────┘     └──────────────────────┘    └──────────────────────┘
```

- **Milestone 1:** Exploratory Data Analysis, text normalization, TF-IDF vectorization, Cosine Similarity baseline, and Majority Class baseline ([`notebooks/milestone-1.ipynb`](notebooks/milestone-1.ipynb)).
- **Milestone 2:** Hugging Face Transformers (`bert-base-uncased`), attention weights & heads, context-aware embeddings (`all-MiniLM-L6-v2`), and Zero-Shot classification with BART-MNLI ([`notebooks/milestone-2.ipynb`](notebooks/milestone-2.ipynb)).
- **Milestone 3:** Dense retrieval with FAISS, Cross-Encoder reranking, RAG context budgeting, adversarial context resilience testing, and Hit Rate @ 5 evaluation ([`notebooks/milestone-3.ipynb`](notebooks/milestone-3.ipynb)).

---

## DeBERTa-v3 Training Details & Curves

### Hyperparameter Configuration

| Hyperparameter | Value | Description |
| :--- | :--- | :--- |
| **Model Backbone** | `microsoft/deberta-v3-base` | 183.8M parameters with disentangled attention |
| **Max Sequence Length** | `320 tokens` | Truncation & padding for prompt-option pairs |
| **Effective Batch Size** | `16` | Per-device batch size 2 $\times$ Gradient Accumulation 8 |
| **Base Learning Rate** | `1e-5` | AdamW optimizer ($\beta_1=0.9, \beta_2=0.999$) |
| **Weight Decay** | `0.01` | Regularization on non-bias weights |
| **LR Scheduler** | Cosine with Warmup | 10% warmup steps over total training iterations |
| **Mixed Precision** | PyTorch AMP (`float16`) | Accelerated training and reduced VRAM footprint |
| **Gradient Clipping** | Norm 1.0 | Safeguards against gradient explosions |
| **Epochs** | `7` | Checkpointing best validation MAP@3 score |

### Progression Across Epochs

| Epoch | Training Loss | Train Acc | Val Accuracy | Val Macro F1 | Val MAP@3 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 1.5720 | 0.2556 | 0.6750 | 0.6764 | 0.7967 |
| **2** | 0.6561 | 0.7589 | 0.9900 | 0.9904 | 0.9917 |
| **3** | 0.2318 | 0.9172 | 0.9950 | 0.9959 | 0.9967 |
| **4** | 0.0913 | 0.9700 | **1.0000** | **1.0000** | **1.0000** |
| **5** | 0.0470 | 0.9828 | **1.0000** | **1.0000** | **1.0000** |
| **6** | 0.0281 | 0.9922 | **1.0000** | **1.0000** | **1.0000** |
| **7** | 0.0178 | 0.9950 | **1.0000** | **1.0000** | **1.0000** |

---

## Streamlit Web Application Overview

The repository features an interactive web application implemented in [`app.py`](app.py):

```bash
streamlit run app.py
```

### Application Features

1. **Single Question Solver**:
   - Test benchmark physics questions (Supersymmetric Quantum Mechanics, JWST Redshift, Landau-Lifshitz-Gilbert, Maxwell's Demon) or input custom prompts and options.
   - Dynamic RAG toggle: Retrieves context from `data/wikipedia_pages` or queries MediaWiki live.
   - Model selector: Instant switching between DeBERTa-v3, MiniLM SentenceTransformer, BART-MNLI Zero-Shot, or TF-IDF.
   - Interactive probability distribution bar chart and formatted MAP@3 top-3 recommendation.

2. **Batch Test Set Predictor**:
   - Upload any custom CSV matching the schema `id, prompt, A, B, C, D, E` or load the repository test dataset (`data/test.csv`, 500 rows).
   - Real-time progress bar with per-row processing indicators.
   - Generates compliant MAP@3 top-3 ranking strings and provides instant download of [`submission.csv`](submission.csv).

3. **Model & Training Architecture**:
   - Comprehensive breakdown of transformer hyperparameters and optimization configs.
   - Interactive charts visualizing validation accuracy, MAP@3, and training loss progression across epochs.

4. **Dataset Explorer & Metrics**:
   - Searchable, paginated tables for training (`data/train.csv`) and test (`data/test.csv`) sets with live keyword filtering.
   - LaTeX mathematical definition of the MAP@3 evaluation metric.

---

## Repository Structure

```
DL-GenAI-Project-t22026/
├── .devcontainer/                         # Containerized development configuration
├── data/
│   ├── sample_submission.csv              # Baseline submission template
│   ├── test.csv                           # 500 test questions (id, prompt, A-E)
│   ├── test_keywords.csv                  # Extracted topic keywords for retrieval
│   ├── train.csv                          # 2,000 training questions with answers
│   └── wikipedia_pages/                   # Pre-scraped plain-text Wikipedia articles
├── models/                                # Model weights & checkpoints directory
├── notebooks/
│   ├── dl-24f2001637-notebook-t22026 - V1.ipynb # Technique 1: Classical ML & Custom Bi-LSTM (Scratch)
│   ├── dl-24f2001637-notebook-t22026 - V2.ipynb # Technique 2: DeBERTa-v3 Multiple-Choice Fine-Tuning
│   ├── dl-24f2001637-notebook-t22026 - V3.ipynb # Technique 3: Dense RAG (FAISS + Cross-Encoder + LLaMA-3)
│   ├── milestone-1.ipynb                  # Milestone 1: EDA, TF-IDF, Cosine & Baselines
│   ├── milestone-2.ipynb                  # Milestone 2: Transformers, Attention, Sentence-BERT
│   ├── milestone-3.ipynb                  # Milestone 3: FAISS, Dense RAG & Adversarial Tests
│   ├── eda.ipynb                          # Comprehensive Exploratory Data Analysis
│   └── dl-24f2001637-notebook-t22026 - V*.ipynb # Additional development iterations
├── reports/
│   └── 24f2001637_DG_T22026.pdf           # Milestone evaluation project report
├── src/
│   └── scrape_wikipedia.py                # Automated Wikipedia search & scraper utility
├── app.py                                 # Interactive Streamlit application
├── DL-24f2001637-notebook-t22026.ipynb    # Final complete training & inference notebook
├── requirements.txt                       # Project dependencies
├── submission.csv                         # Final test predictions formatted for MAP@3
└── README.md                              # Repository documentation
```

---

## Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/24f2001637/DL-GenAI-Project-t22026.git
cd DL-GenAI-Project-t22026
```

### 2. Set Up Virtual Environment
```bash
# Linux / macOS
python3 -m venv venv
source venv/bin/activate

# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Launch the Interactive Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### 5. (Optional) Run the Wikipedia Scraper Utility
To re-scrape or expand the local scientific knowledge base:
```bash
python src/scrape_wikipedia.py --csv data/test_keywords.csv --out data/wikipedia_pages --delay 0.5 --resume -v
```

---

## Git Workflow & Branch Compliance

In accordance with course evaluation guidelines, development was strictly partitioned across dedicated milestone branches:

| Branch Name | Primary Focus | Status |
| :--- | :--- | :--- |
| [`main`](https://github.com/24f2001637/DL-GenAI-Project-t22026/tree/main) | Production code, final notebook, Streamlit app, and report | **Active / Stable** |
| [`milestone-1`](https://github.com/24f2001637/DL-GenAI-Project-t22026/tree/milestone-1) | Milestone 1 EDA, TF-IDF baseline, and MAP@3 implementation | **Preserved for Audit** |
| [`milestone-2`](https://github.com/24f2001637/DL-GenAI-Project-t22026/tree/milestone-2) | Transformer probing, sentence embeddings, and zero-shot NLI | **Preserved for Audit** |
| [`milestone-3`](https://github.com/24f2001637/DL-GenAI-Project-t22026/tree/milestone-3) | FAISS dense retrieval, RAG pipeline, and adversarial tests | **Preserved for Audit** |

> **Audit Note:** All milestone branches are permanently preserved in the remote repository for academic grading and verification.

---

## Submission Summary

| Field | Value |
| :--- | :--- |
| **Student Roll Number** | `24f2001637` |
| **Program** | IIT Madras BS Degree in Data Science and Applications |
| **Course** | Deep Learning & Generative AI |
| **Term** | Term 2 2026 (T2 2026) |
| **Best Model** | `microsoft/deberta-v3-base` (Multiple Choice Sequence-Pair) |
| **Evaluation Metric** | MAP@3 (Mean Average Precision at Rank 3) |
| **Kaggle Public Score** | **0.75976 MAP@3** |
| **Predictions File** | [`submission.csv`](submission.csv) (500 rows, Top-3 ranked options) |

---

## License & Acknowledgements

This project was developed as part of the academic curriculum for the IIT Madras BS Degree in Data Science and Applications. Model backbones and tokenizer weights are provided courtesy of Hugging Face, Microsoft Research, BAAI, and Meta AI.
