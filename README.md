## 📌Preliminary


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


#### Option 2: Using conda (Original)

```shell
# create a new environment
conda create -n graphrouter python=3.10
conda activate graphrouter

# install pytorch. Modify the command to align with your own CUDA version.
pip3 install torch  --index-url https://download.pytorch.org/whl/cu118

# install related libraries
pip install -r requirements.txt
```

### Dataset Preparation

First, generate 'data/unified_qa_data.csv'.

```bash
python data_processing/multidata_unify.py
```
Then, generate `data/router_data.csv` and `configs/llm_description_embedding.pkl` by setting your api_key in `configs/config.yaml`.

```bash
python data_processing/construct_router_data.py
```

For your convenience, we provide download links for the 'unified_qa_data.csv' and 'router_data.csv' files we generated. Please download them and put them in `data` folder.

[unified_qa_data.csv](https://drive.google.com/file/d/1__SY7UScvX1xPWeX1NK6ZulLMdZTqBcI/view?usp=share_link)
[router_data.csv](https://drive.google.com/file/d/1YYn-BV-5s2amh6mKLqKMR0H__JB-CKU4/view?usp=share_link)

## ⭐Experiments

### Training and Evaluation

Run experiments and print/save evaluation results on metrics Performance, Cost, and Reward. You can edit the hyperparameters in `configs/config.yaml` or using your own config_file.

```bash
python run_exp.py --config_file [config]
```

## 🔍Inference

### Running Inference on Existing Queries

Use `inference.py` to run inference on queries from the existing dataset:

```bash
# Run inference on a specific query ID (default: 0)
python inference.py --query_id 0

# Run inference on query ID 5
python inference.py --query_id 5
```

**Arguments:**
- `--query_id`: Query index from the dataset (default: 0)

### Running Inference on New Queries

Use `inference_withQuery.py` to run inference on your own custom queries:

```bash
# Run inference with a custom query
python inference_withQuery.py --input_text "What is the capital of France?" --task_id "alpaca_data"

# Run inference with another custom query
python inference_withQuery.py --input_text "Explain quantum computing" --task_id "alpaca_data"
```

**Arguments:**
- `--input_text`: Your custom query text (default: "Name three planets in our solar system that have rings")
- `--task_id`: Task identifier from the dataset (default: "alpaca_data")

**Available task IDs:**
- `"alpaca_data"` - General instruction following
- `"GSM8K"` - Math word problems
- `"multi_news"` - News summarization
- `"SQUAD"` - Question answering

### Understanding the Output

Both inference scripts will:
1. Load the trained GraphRouter model
2. Process your query through the graph neural network
3. Return the best LLM recommendation and scores for all available LLMs
4. Display an interactive plot showing LLM scores across different scenarios

The output includes:
- **Best LLM**: The recommended LLM for your query
- **Scores**: Confidence scores for all available LLMs
- **Visualization**: Interactive bar chart showing LLM performance

### Tricks for Adapting GraphRouter to Other Tasks and Datasets

1. **Embedding Normalization**  
   - Check whether input embeddings are normalized.  
   - On some datasets, skipping normalization leads to suboptimal results.  

2. **Network Initialization**  
   - Experiment with different initialization methods.  
   - Try varying random seeds or using alternative initialization schemes.  

3. **Model Saving Strategy**  
   - Instead of saving models based on highest accuracy, save checkpoints with the best evaluation set performance.  
   - This can yield better results on certain tasks.  

4. **Learning Rate Tuning**  
   - Adjust learning rate carefully.  
   - Slightly increasing it may help avoid local optima and improve stability.  

## Citation

```bibtex
@inproceedings{feng2024graphrouter,
  title={Graphrouter: A graph-based router for llm selections},
  author={Feng, Tao and Shen, Yanzhen and You, Jiaxuan},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2024}
}
```
