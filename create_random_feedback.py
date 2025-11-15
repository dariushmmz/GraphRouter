import os
import re
import json
import yaml
import torch
import numpy as np
import pandas as pd

import json
import pickle
import plotly.io as pio

from model.graph_nn import EncoderDecoderNet, form_data

device = "cpu"  # use CPU by default


llm_embedding_path = 'configs/llm_description_embedding.pkl'
llm_description_path = 'configs/LLM_Descriptions.json'

embedding_dim = 16
# edge_dim = 3


def loadpkl(filename: str) -> any:
# from data_processing.utils import loadjson, loadpkl

    """
    Load data from a pickle file.

    Args:
        filename: Path to the pickle file

    Returns:
        The unpickled object
    """
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    return data

# File I/O functions
def loadjson(filename: str) -> dict:
    """
    Load data from a JSON file.

    Args:
        filename: Path to the JSON file

    Returns:
        Dictionary containing the loaded JSON data
    """
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


def build_single_query_datasafe(df, llm_embeddings, query_id, scenario):
    llm_desc = loadjson(llm_description_path)
    llm_names = list(llm_desc.keys())
    num_llms = len(llm_names)
    nrows = len(df)
    if nrows % num_llms != 0:
        raise ValueError(f"router_data rows {nrows} not divisible by num_llms {num_llms}")

    df = df.iloc[[int(query_id)]]

    query_embedding_list_raw = df['query_embedding'].tolist()
    task_embedding_list_raw = df['task_description_embedding'].tolist()

    query_embedding_list = []
    task_embedding_list = []

    for inter in query_embedding_list_raw:
        inter = re.sub(r'\s+', ', ', inter.strip())
        try:
            inter = json.loads(inter)
        except:
            inter = inter.replace("[[,", "[[")
            inter = json.loads(inter)
        query_embedding_list.append(inter[0])

    for inter in task_embedding_list_raw:
        inter = re.sub(r'\s+', ', ', inter.strip())
        try:
            inter = json.loads(inter)
        except:
            inter = inter.replace("[[,", "[[")
            inter = json.loads(inter)
        task_embedding_list.append(inter[0])


    q_emb = np.array(query_embedding_list)
    t_emb = np.array(task_embedding_list)

    q_emb = ensure_2d(parse_embedding_field(q_emb[0]))
    t_emb = ensure_2d(parse_embedding_field(t_emb[0]))

    llm_embeddings = np.asarray(llm_embeddings, dtype=float)
    if llm_embeddings.ndim == 1:
        raise ValueError("llm_embeddings is 1-D; expected (num_llms, dim)")
    if llm_embeddings.shape[0] != num_llms and llm_embeddings.shape[0] != 1:
        raise ValueError(f"llm_embeddings first dim {llm_embeddings.shape[0]} != num_llms {num_llms}")

    if llm_embeddings.shape[0] == 1 and num_llms > 1:
        llm_embeddings = np.tile(llm_embeddings, (num_llms, 1))

    effect_list = np.array(df['effect'].tolist(), dtype=float)
    cost_list = np.array(df['cost'].tolist(), dtype=float)

    # ====================== Build combined_edge ======================
    # include feedback (normalized)
    if 'avg_feedback' in df.columns:
        feedback_list = np.array(df['avg_feedback'].fillna(0).tolist())
        # normalize if values in 1..5
        if feedback_list.max() > 1.0:
            feedback_list = (feedback_list - 1.0) / 4.0

        
        combined_edge = np.concatenate(
            (
                cost_list.reshape(-1, 1),
                effect_list.reshape(-1, 1),
                feedback_list.reshape(-1, 1)
            ),
            axis=1
        )
    else:
        combined_edge = np.concatenate(
            (
                cost_list.reshape(-1, 1),
                effect_list.reshape(-1, 1),
            ),
            axis=1
        )

    # combined_edge = np.concatenate([cost_list.reshape(-1, 1), effect_list.reshape(-1, 1)], axis=1)

    if scenario == "Performance First":
        eff_adj = 1.0 * effect_list - 0.0 * cost_list
    elif scenario == "Balance":
        eff_adj = 0.5 * effect_list - 0.5 * cost_list
    else:
        eff_adj = 0.2 * effect_list - 0.8 * cost_list

    label = np.eye(num_llms)[np.argmax(eff_adj)].reshape(-1, 1)

    org_node = [0] * num_llms
    des_node = list(range(num_llms))
    mask_all = np.ones(num_llms, dtype=bool)

    # # pad/truncate combined_edge
    # if combined_edge.shape[1] != edge_dim:
    #     if combined_edge.shape[1] < edge_dim:
    #         pad = np.zeros((combined_edge.shape[0], edge_dim - combined_edge.shape[1]), dtype=float)
    #         combined_edge = np.concatenate([combined_edge, pad], axis=1)
    #     else:
    #         combined_edge = combined_edge[:, :edge_dim]
    
    data_dict = {
        "task_id": t_emb.astype(np.float32),
        "query_feature": q_emb.astype(np.float32),
        "llm_feature": llm_embeddings.astype(np.float32),
        "org_node": org_node,
        "des_node": des_node,
        "edge_feature": eff_adj.astype(np.float32),
        "label": label.astype(np.float32),
        "edge_mask": mask_all,
        "combined_edge": combined_edge.astype(np.float32),
        "train_mask": mask_all,
        "valide_mask": mask_all,
        "test_mask": mask_all
    }

    query_text = df['query'].iloc[0] if 'query' in df.columns else f"query_{query_id}"
    task_text = df['task_description'].iloc[0] if 'task_description' in df.columns else f"task_{query_id}"

    return data_dict, (query_text, task_text), llm_names


def run_safe_inference(query_id=0, checkpoint=None, scenario="Cost First"):
    df = pd.read_csv(saved_router_data_path)
    llm_embeddings = loadpkl(llm_embedding_path)

    data_dict, (query_text, task_text), llm_names = build_single_query_datasafe(df, llm_embeddings, query_id, scenario)

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
    # in_edges = edge_dim  # use the same edge_dim as training
    # print(in_edges)


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
    # Get top 3 indices and scores
    # topk_values, topk_indices = torch.topk(pred, k=3, dim=1)
    # topk_indices = topk_indices[0].cpu().tolist()
    # topk_scores = topk_values[0].cpu().tolist()
    # top3 = [(llm_names[i], float(score)) for i, score in zip(topk_indices, topk_scores)]

    scores = {llm_names[i]: float(pred[0, i].cpu().item()) for i in range(len(llm_names))}

    return {
        "query_id": int(query_id),
        "query_text": query_text,
        "task_text": task_text,
        "best_llm": llm_names[best_idx],
        "scores": scores
    }

import plotly.express as px
import pandas as pd

def plot_llm_scores(query_id, scenarios=None, plot=True):
    """
    Run LLM inference for a given query and plot scores across scenarios.

    Args:
        query_id (int): The query index to run inference on.
        scenarios (list, optional): List of scenario names. Defaults to ["Cost First", "Balance", "Performance First"].

    Returns:
        pd.DataFrame: DataFrame containing LLM scores for all scenarios.
        plotly.graph_objs._figure.Figure: The Plotly figure object.
    """
    if scenarios is None:
        scenarios = ["Cost First", "Balance", "Performance First"]

    results_list = []
    best_selected_llms = []

    # Run inference for each scenario and prepare results
    for i, scenario in enumerate(scenarios):
        out = run_safe_inference(query_id=query_id, scenario=scenario)
        if i == 0:
            print("Query id:", out["query_id"], "- Query:", out['query_text'])
            print()
        print(f"-------------{scenario}-------------------")
        print("Best LLM:", out['best_llm'])
        best_selected_llms.append(out['best_llm'])


        for llm, score in out['scores'].items():
            results_list.append({"LLM": llm, "Score": score, "Scenario": scenario})
    print('\n\n')

    df_scores = pd.DataFrame(results_list)


    if plot:
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
            text=df_scores["Score"].apply(lambda x: f"{x:.3f}"),
            title=f"LLM Scores Across Different Scenarios for Query {query_id}"
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

        os.makedirs("inference_results", exist_ok=True)
        # Save as HTML file (recommended for headless environments)
        fig.write_html(f"inference_results/llm_scores_query_{query_id}.html")
        print(f"Graph saved as: llm_scores_query_{query_id}.html")
        
        pio.renderers.default = "browser"   # open in default browser
        fig.show()
    
    return best_selected_llms

import json, time, os

FEEDBACK_PATH = "data/feedback.jsonl"

def save_feedback(query_id, llm_name, user_score, extra=None):
    rec = {
        "timestamp": time.time(),
        "query_id": int(query_id),
        "LLM": llm_name,
        "Score": float(user_score),
    }
    if extra:
        rec.update(extra)
    os.makedirs(os.path.dirname(FEEDBACK_PATH), exist_ok=True)
    with open(FEEDBACK_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def ask_and_save_feedback(query_id, predicted_llm):
    s = input("\n===========> Score 1..5 (or blank to reject): ").strip()
    if s.isdigit():
        score = int(s)
        save_feedback(query_id, predicted_llm, score)


import random

def auto_score_feedback(query_id, predicted_llms):
    """
    Assign random feedback score, but penalize LLaMA-3 (70b) predictions.
    """
    for predicted_llm in predicted_llms:
        if predicted_llm.strip().lower() == "NousResearch".lower():
            score = random.randint(1, 2)  # penalize
        else:
            score = random.randint(3, 5)  # normal range
        save_feedback(query_id, predicted_llm, score, extra={"auto": True})
        print(f"[Feedback Saved] Query {query_id} | {predicted_llm} → Score: {score}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="Run all queries automatically")
    parser.add_argument("--query_id", type=int, default=None)
    args = parser.parse_args()

    saved_router_data_path = "data/router_data_with_feedback.csv"
    # saved_router_data_path = "data/router_data.csv"

    # model_path = 'model_path/best_model_qa.pth'
    model_path = 'model_path/model_with_feedback.pth'


    df = pd.read_csv(saved_router_data_path)

    if args.all:
        # num_queries = len(df) // len(loadjson(llm_description_path))
        num_queries = len(df) 
        print(f"Running inference for {num_queries} queries...")
        for qid in range(num_queries):
            # predicted_llm = plot_llm_scores(query_id=qid, plot=False)
            # auto_score_feedback(qid, predicted_llm)

            try:
                predicted_llms = plot_llm_scores(query_id=qid, plot=False)
                auto_score_feedback(qid, predicted_llms)
            except Exception as e:
                print(f"⚠️ Skipped query {qid} due to error: {e}")
    else:
        qid = args.query_id if args.query_id is not None else 0
        predicted_llm = plot_llm_scores(query_id=qid)
        auto_score_feedback(qid, predicted_llm)
