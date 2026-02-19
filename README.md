## 📌 Preliminary

**GraphRouter** is a graph-based router for selecting among multiple LLMs per query. The model learns to route queries to the best LLM under configurable scenarios (**Performance First**, **Balance**, **Cost First**) using a GNN over query–LLM edges with task and cost/effect features.

### Environment Setup

#### Option 1: Using uv (Recommended)

```shell
# Install uv if you haven't already
pip install uv

# Create and activate virtual environment with uv
uv venv
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
uv sync
```

#### Option 2: Using pip / pyproject.toml

```shell
pip install -e .
```

#### Option 3: Using conda

```shell
conda create -n graphrouter python=3.10
conda activate graphrouter

# Install PyTorch (modify for your CUDA version)
pip3 install torch --index-url https://download.pytorch.org/whl/cu118

# Install project
pip install -e .
```

Set API keys for LLM providers (Together, OpenRouter, etc.) in `.env` or in `configs/config.yaml` as needed for data construction and training.

---

## Dataset Preparation

Pipeline order:

1. **Unify data** → `unified_data.csv`
2. **Build router data** (LLM calls, rewards, costs) → `router_data.csv`
3. **(Optional) Add OMS metrics** → updates `router_data.csv` with RQS/OMS and overwrites `effect`

### Step 1: Generate `unified_data.csv`

Set `data_dir` in `configs/config.yaml` to your dataset folder (e.g. `data/GSM8K`). The script uses the last segment of `data_dir` as the task name (e.g. `GSM8K`).

```bash
python data_processing/multidata_unify.py
```

Output: `{data_dir}/unified_data.csv` (e.g. `data/GSM8K/unified_data.csv`).

### Step 2: Generate `router_data.csv`

Uses `unified_data.csv`, calls each configured LLM per query, and computes reward and cost. Expects `LLM_Descriptions.json` (e.g. under `configs/` or your data folder). Set `data_dir` and API keys (e.g. in `.env`: `api_key`, `openrouter_api_key`).

```bash
python data_processing/construct_router_data.py
```

Output: `{data_dir}/router_data.csv` and `configs/llm_description_embedding.pkl`. If using feedback, router data can be under `{data_dir}/feedback/router_data.csv`.

### Step 3 (Optional): Add OMS metrics

Adds RQS (reasoning quality) and OMS (exact match + reasoning) and overwrites `effect` in the router CSV. Edit the `data_dir` list at the bottom of the script to point to your `router_data.csv` path.

```bash
python data_processing/oms_metric.py
```

### Pre-built data

For convenience, pre-generated files can be downloaded and placed in the `data` folder:

- [unified_qa_data.csv](https://drive.google.com/file/d/1bOfJkAm3nflRz-8y9q6hMkZGd3DPnbMc/view?usp=sharing)
- [router_data.csv](https://drive.google.com/file/d/1lM0bhVpVcztLBNAgKpR1HtAB_xT93A-k/view?usp=sharing)

---

## ⭐ Experiments

### Training

Run training with `configs/config.yaml` (or your own config):

```bash
python run_exp.py --config_file configs/config.yaml
```

Set `data_dir` to the directory containing `router_data.csv` (or `feedback/router_data.csv` when `feedback: true`). Training uses Weights & Biases when `wandb_key` is set (in config or env). Key options in `configs/config.yaml`: `model_path`, `scenario`, `llm_num`, `embedding_dim`, `edge_dim`, `split_ratio`, `train_epoch`, `learning_rate`, `batch_size`, etc.

### Config overview (`configs/config.yaml`)

| Parameter | Description |
|-----------|-------------|
| `data_dir` | Dataset root (e.g. `data/GSM8K`). |
| `query_response_length` | Max response length for LLM calls. |
| `model_path` | Path to trained model for inference. |
| `feedback` | If true, use `feedback/router_data.csv` when present. |
| `scenario` | `Performance First`, `Balance`, or `Cost First`. |
| `llm_num` | Number of LLMs (must match data and LLM_Descriptions). |
| `embedding_dim`, `edge_dim` | GNN hidden and edge feature dimensions. |
| `split_ratio` | Train/val/test ratio, e.g. `[0.7, 0.1, 0.2]`. |
| `train_epoch`, `learning_rate`, `weight_decay`, `batch_size` | Training hyperparameters. |

---

## 🔍 Inference

### Running inference on a query from the dataset

Use `inference.py` to run the trained router on a **single query index** from `router_data.csv`:

```bash
# Query index 0 (default)
python inference.py --config_file configs/config.yaml --id 0

# Query index 5
python inference.py --config_file configs/config.yaml --id 5
```

**Arguments:**

- `--config_file`: Path to YAML config (default: `configs/config.yaml`).
- `--id`: Query index (integer). The script uses the block of rows for that query (one row per LLM).
- `--adaptive_updater`: If set, runs the adaptive feedback updater after inference.

Ensure `model_path`, `data_dir`, `scenario`, `llm_num`, `embedding_dim`, and `edge_dim` in the config match the trained model.

### Output

The script will:

1. Load the trained GraphRouter model from `model_path`.
2. Run inference for the given query and print **Top-3 predicted LLMs** and **Top-3 ground-truth LLMs** with scores.
3. Save an interactive HTML plot under `data/results/{task_id}/llm_scores_query_{id}.html`.

---

## Project structure (main entries)

```
configs/
  config.yaml                  # Main config
  LLM_Descriptions.json         # LLM names and descriptions
  llm_description_embedding.pkl # Precomputed LLM embeddings

data_processing/
  multidata_unify.py           # Step 1: build unified_data.csv
  construct_router_data.py     # Step 2: build router_data.csv
  oms_metric.py                # Step 3 (optional): add OMS metrics
  llm_engine.py                # LLM API and evaluation
  utils.py                     # Embeddings, I/O, parsing
  instructions.py              # Task instructions (e.g. MATH, GSM8K)
  delayed_reward.py            # Adaptive feedback updater

model/
  multi_task_graph_router.py   # Router: training + infer_single_query()
  graph_nn.py                  # GNN (EncoderDecoderNet, form_data)

inference.py                   # Run inference for one query (--id)
run_exp.py                     # Run training
```

---

## Tricks for adapting GraphRouter to other tasks and datasets

1. **Embedding normalization**  
   Check whether input embeddings are normalized; on some datasets, skipping normalization can hurt performance.

2. **Network initialization**  
   Try different seeds or initialization schemes.

3. **Model saving strategy**  
   Save checkpoints by best validation performance rather than only accuracy when that works better for your task.

4. **Learning rate**  
   Tune learning rate; a slightly higher value can help avoid local optima and improve stability.

---

## Citation

```bibtex
@inproceedings{feng2024graphrouter,
  title={Graphrouter: A graph-based router for llm selections},
  author={Feng, Tao and Shen, Yanzhen and You, Jiaxuan},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2024}
}
```

- **Repository**: [https://github.com/ulab-uiuc/GraphRouter](https://github.com/ulab-uiuc/GraphRouter)  
- **Documentation**: [https://ulab-uiuc.github.io/GraphRouter/](https://ulab-uiuc.github.io/GraphRouter/)
