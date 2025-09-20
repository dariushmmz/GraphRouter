import os
import re
import json
import yaml
import torch
import numpy as np
import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer  # Required for get_embedding
import plotly.io as pio

# Assume utils.py is in the same directory or path; import necessary functions
# The user attached utils.py, so this assumes it's available
from data_processing.utils import get_embedding, model_prompting  # Import from attached utils.py
from model.graph_nn import EncoderDecoderNet, form_data

device = "cpu"  # Use CPU by default

saved_router_data_path = "data/router_data.csv"
llm_embedding_path = 'configs/llm_description_embedding.pkl'
llm_description_path = 'configs/LLM_Descriptions.json'
embedding_dim = 16
edge_dim = 3
model_path = 'model_path/best_model_qa.pth'


def loadpkl(filename: str) -> any:
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    return data

def loadjson(filename: str) -> dict:
    with open(filename, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def parse_embedding_field(raw):
    if isinstance(raw, (list, np.ndarray)):
        return np.array(raw, dtype=float)
    s = str(raw).strip()
    s = re.sub(r'\s+', ', ', s)
    try:
        parsed = json.loads(s)
    except Exception:
        parsed = json.loads(s.replace("[[,", "[["))
    return np.array(parsed[0], dtype=float)

def ensure_2d(arr):
    arr = np.asarray(arr)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    return arr

def build_new_query_data(task_id, query_text, scenario):
    df = pd.read_csv(saved_router_data_path)
    if task_id not in df['task_id'].unique():
        raise ValueError(f"Task '{task_id}' not found in existing data. Choose from {df['task_id'].unique()} or add new task following README instructions.")

    # Get task details from an existing row
    row = df[df['task_id'] == task_id].iloc[0]
    task_description = row['task_description']
    t_emb = parse_embedding_field(row['task_description_embedding'])

    # Generate embedding for the new query
    q_emb = get_embedding([query_text])

    llm_desc = loadjson(llm_description_path)
    llm_names = list(llm_desc.keys())
    num_llms = len(llm_names)
    llm_embeddings = loadpkl(llm_embedding_path)

    # For new query, set dummy/zero values for edges (predictions will be based on embeddings)
    effect_list = np.zeros(num_llms)
    cost_list = np.zeros(num_llms)
    combined_edge = np.concatenate([cost_list.reshape(-1, 1), effect_list.reshape(-1, 1)], axis=1)

    if scenario == "Performance First":
        eff_adj = 1.0 * effect_list - 0.0 * cost_list
    elif scenario == "Balance":
        eff_adj = 0.5 * effect_list - 0.5 * cost_list
    else:
        eff_adj = 0.2 * effect_list - 0.8 * cost_list

    label = np.eye(num_llms)[0].reshape(-1, 1)  # Dummy label, not used in inference

    org_node = [0] * num_llms
    des_node = list(range(num_llms))
    mask_all = np.ones(num_llms, dtype=bool)

    # Pad/truncate combined_edge to match edge_dim
    if combined_edge.shape[1] != edge_dim:
        if combined_edge.shape[1] < edge_dim:
            pad = np.zeros((combined_edge.shape[0], edge_dim - combined_edge.shape[1]), dtype=float)
            combined_edge = np.concatenate([combined_edge, pad], axis=1)
        else:
            combined_edge = combined_edge[:, :edge_dim]

    data_dict = {
        "task_id": ensure_2d(t_emb).astype(np.float32),
        "query_feature": ensure_2d(q_emb).astype(np.float32),
        "llm_feature": ensure_2d(llm_embeddings).astype(np.float32),
        "org_node": org_node,
        "des_node": des_node,
        "edge_feature": eff_adj.astype(np.float32),
        "label": label.astype(np.float32),
        "edge_mask": mask_all,
        "combined_edge": combined_edge.astype(np.float32),
        "train_mask": np.zeros(num_llms, dtype=bool),  # No training for new query
        "valide_mask": np.zeros(num_llms, dtype=bool),
        "test_mask": mask_all  # Treat as test
    }

    return data_dict, (query_text, task_description), llm_names

def run_new_query_inference(task_id, query_text, checkpoint=None, scenario="Balance"):
    data_dict, (query_text, task_text), llm_names = build_new_query_data(task_id, query_text, scenario)

    form = form_data("cpu")
    data_obj = form.formulation(
        task_id=data_dict['task_id'],
        query_feature=data_dict['query_feature'],
        llm_feature=data_dict['llm_feature'],
        org_node=data_dict['org_node'],
        des_node=data_dict['des_node'],
        edge_feature=data_dict['edge_feature'],
        label=data_dict['label'],
        edge_mask=data_dict['edge_mask'],
        combined_edge=data_dict['combined_edge'],
        train_mask=data_dict['train_mask'],
        valide_mask=data_dict['valide_mask'],
        test_mask=data_dict['test_mask']
    )

    edge_mask_t = torch.tensor(data_obj.edge_mask, dtype=torch.bool)
    edge_can_see = torch.tensor(data_obj.test_mask, dtype=torch.bool)

    q_dim = data_obj.query_features.shape[1]
    llm_dim = data_obj.llm_features.shape[1]
    in_edges = data_obj.combined_edge.shape[1]

    model = EncoderDecoderNet(query_feature_dim=q_dim, llm_feature_dim=llm_dim,
                              hidden_features=embedding_dim, in_edges=in_edges).to(device)

    ckpt = checkpoint or model_path
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    with torch.no_grad():
        pred = model(task_id=data_obj.task_id,
                     query_features=data_obj.query_features,
                     llm_features=data_obj.llm_features,
                     edge_index=data_obj.edge_index,
                     edge_mask=edge_mask_t,
                     edge_can_see=edge_can_see,
                     edge_weight=data_obj.combined_edge)

    pred = pred.reshape(-1, len(llm_names))

    best_idx = int(torch.argmax(pred, dim=1).cpu().item())
    scores = {llm_names[i]: float(pred[0, i].cpu().item()) for i in range(len(llm_names))}

    return {
        "query_text": query_text,
        "task_text": task_text,
        "best_llm": llm_names[best_idx],
        "scores": scores
    }

import plotly.express as px
import pandas as pd
from datetime import datetime

def plot_llm_scores(query_text, task_id, scenarios="Balance"):
    """
    Run LLM inference for a given query and plot scores across scenarios.

    Args:
        query_text (str): The text of the new query to run inference on.
        task_id (str): The task ID associated with the query (e.g., "Alpaca").
        scenarios (list, optional): List of scenario names. Defaults to ["Cost First", "Balance", "Performance First"].

    Returns:
        pd.DataFrame: DataFrame containing LLM scores for all scenarios.
        plotly.graph_objs._figure.Figure: The Plotly figure object.
    """
    scenario = "Balance"
    results_list = []

    out = run_new_query_inference(task_id=task_id, query_text=query_text, scenario=scenario)
    print("Query Text:", out["query_text"])
    print("Task:", out["task_text"])
    print("Best LLM:", out['best_llm'])
    print()

    for llm, score in out['scores'].items():
        results_list.append({"LLM": llm, "Score": score, "Scenario": scenario})

    df_scores = pd.DataFrame(results_list)

    # Compute min/max for y-axis with margin
    y_min = df_scores["Score"].min()
    y_max = df_scores["Score"].max()
    margin = (y_max - y_min) * 0.05  # 5% margin

    # Create interactive grouped bar chart
    fig = px.bar(
        df_scores,
        x="LLM",
        y="Score",
        color="Scenario",
        barmode="group",
        # text=df_scores["Score"].apply(lambda x: f"{x:.3f}"),
        title=f"LLM Scores Across Different Scenarios for Query '{query_text}'"
    )

    # Update layout for better readability
    fig.update_layout(
        xaxis_title="LLM",
        yaxis_title="Score",
        xaxis_tickangle=-45,
        yaxis=dict(showgrid=True, range=[y_min - margin, y_max + margin]),
        legend_title="Scenario",
        template="plotly_white"
    )

    safe_query_text = query_text.replace(" ", "_").replace("?", "").replace("!", "")[:50]
    os.makedirs("inference_results", exist_ok=True)
    # Save as HTML file (recommended for headless environments)
    fig.write_html(f"inference_results/llm_scores_query_{safe_query_text}.html")
    print(f"Graph saved as: llm_scores_query_{safe_query_text}.html")
    
    pio.renderers.default = "browser"   # open in default browser
    fig.show()
    
    return df_scores, fig

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_text", type=str, default="Name three planets in our solar system that have rings")
    parser.add_argument("--task_id", type=str, default="alpaca_data")
    args = parser.parse_args()
    input_text = args.input_text
    task_id = args.task_id

    # Run inference
    out = run_new_query_inference(task_id, input_text)
    df_scores, fig = plot_llm_scores(input_text, task_id)