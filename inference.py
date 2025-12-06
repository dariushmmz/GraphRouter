"""
Inference module for GraphRouter LLM selection system.

This module provides functionality for running inference on queries to select
the best LLM based on different scenarios (Cost First, Balance, Performance First).
"""

import argparse
import json
import os
import pickle
import random
import re
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import torch
from scipy.special import softmax

from model.graph_nn import EncoderDecoderNet, form_data


# ============================================================================
# Constants
# ============================================================================

SCENARIOS = ["Cost First", "Balance", "Performance First"]
DEFAULT_SCENARIO = "Cost First"

# Scenario weights: (effect_weight, cost_weight)
SCENARIO_WEIGHTS = {
    "Performance First": (1.0, 0.0),
    "Balance": (0.6, 0.4),
    "Cost First": (0.3, 0.7),
}

# Softmax temperature for score sharpening
SOFTMAX_TEMPERATURE = 0.15
LABEL_SMOOTHING = 0.05

# Feedback score normalization
FEEDBACK_MAX_SCORE = 5.0
FEEDBACK_MIN_SCORE = 1.0
FEEDBACK_NORMALIZED_RANGE = 4.0


# ============================================================================
# Utility Functions
# ============================================================================

def to_bool_tensor(x: Union[torch.Tensor, Any]) -> torch.Tensor:
    """
    Convert input to a boolean tensor.
    
    Args:
        x: Input to convert (tensor or array-like)
        
    Returns:
        Boolean tensor
    """
    if isinstance(x, torch.Tensor):
        return x.detach().clone().bool()
    return torch.tensor(x, dtype=torch.bool)


def load_pickle(filename: str) -> Any:
    """
    Load data from a pickle file.
    
    Args:
        filename: Path to the pickle file
        
    Returns:
        The unpickled object
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        pickle.UnpicklingError: If the file cannot be unpickled
    """
    with open(filename, 'rb') as file:
        return pickle.load(file)


def load_json(filename: str) -> Dict[str, Any]:
    """
    Load data from a JSON file.
    
    Args:
        filename: Path to the JSON file
        
    Returns:
        Dictionary containing the loaded JSON data
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        json.JSONDecodeError: If the file contains invalid JSON
    """
    with open(filename, 'r', encoding='utf-8') as file:
        return json.load(file)


def parse_embedding(raw: Union[str, List, np.ndarray]) -> np.ndarray:
    """
    Extract float values from a string representation of an embedding.
    
    Args:
        raw: Raw embedding data (string, list, or numpy array)
        
    Returns:
        Numpy array of float values
    """
    if isinstance(raw, str):
        # Extract all float-like tokens from the string
        pattern = r'[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+(?:[eE][-+]?\d+)?'
        nums = re.findall(pattern, raw)
        return np.array([float(n) for n in nums], dtype=float)
    
    return np.array(raw, dtype=float)


def parse_embedding_field(raw: Union[str, List, np.ndarray]) -> np.ndarray:
    """
    Parse an embedding field that may be in various formats.
    
    Args:
        raw: Raw embedding data in various formats
        
    Returns:
        Numpy array of float values
        
    Raises:
        ValueError: If the embedding cannot be parsed
    """
    if isinstance(raw, (list, np.ndarray)):
        return np.array(raw, dtype=float)
    
    try:
        return parse_embedding(raw)
    except (ValueError, TypeError):
        # Try to parse as JSON string
        s = str(raw).strip()
        s = re.sub(r'\s+', ', ', s)
        try:
            parsed = json.loads(s)
        except json.JSONDecodeError:
            # Try fixing common JSON formatting issues
            parsed = json.loads(s.replace("[[,", "[["))
        
        if isinstance(parsed, list) and len(parsed) > 0:
            return np.array(parsed[0] if isinstance(parsed[0], list) else parsed, dtype=float)
        return np.array(parsed, dtype=float)


def ensure_2d(arr: np.ndarray) -> np.ndarray:
    """
    Ensure an array is 2-dimensional.
    
    Args:
        arr: Input array (1D or 2D)
        
    Returns:
        2D array (reshaped if necessary)
    """
    arr = np.asarray(arr)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    return arr


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    """
    Normalize scores to [0, 1] range.
    
    Args:
        scores: Array of scores to normalize
        
    Returns:
        Normalized scores
    """
    score_min = scores.min()
    score_max = scores.max()
    if score_max == score_min:
        return np.ones_like(scores) / len(scores)
    return (scores - score_min) / (score_max - score_min)


def compute_scenario_score(
    effect_normalized: np.ndarray,
    cost_normalized: np.ndarray,
    scenario: str
) -> np.ndarray:
    """
    Compute weighted score based on scenario.
    
    Args:
        effect_normalized: Normalized effect scores
        cost_normalized: Normalized cost scores
        scenario: Scenario name
        
    Returns:
        Combined score array
    """
    effect_weight, cost_weight = SCENARIO_WEIGHTS.get(
        scenario, SCENARIO_WEIGHTS[DEFAULT_SCENARIO]
    )
    return effect_weight * effect_normalized - cost_weight * cost_normalized


# ============================================================================
# Data Preparation Functions
# ============================================================================

def build_single_query_data(
    df: pd.DataFrame,
    llm_embeddings: np.ndarray,
    query_id: int,
    scenario: str,
    llm_description_path: str,
    semantic_embedding_path: Optional[str] = None
) -> Tuple[Dict[str, Any], Tuple[str, str], List[str]]:
    """
    Build data dictionary for a single query inference.
    
    Args:
        df: DataFrame containing router data
        llm_embeddings: Array of LLM embeddings
        query_id: Query identifier
        scenario: Scenario name for scoring
        llm_description_path: Path to LLM descriptions JSON file
        semantic_embedding_path: Optional path to semantic embeddings pickle file
        
    Returns:
        Tuple containing:
            - data_dict: Dictionary with all data for model inference
            - (query_text, task_text): Text descriptions
            - llm_names: List of LLM names
            
    Raises:
        ValueError: If data dimensions are invalid
    """
    llm_desc = load_json(llm_description_path)
    llm_names = list(llm_desc.keys())
    num_llms = len(llm_names)
    nrows = len(df)
    
    if nrows % num_llms != 0:
        raise ValueError(
            f"Router data rows ({nrows}) not divisible by num_llms ({num_llms})"
        )
    
    # Extract rows for this query
    start = int(query_id) * num_llms
    rows = df.iloc[start:start + num_llms]
    
    # Parse embeddings
    q_emb = ensure_2d(parse_embedding_field(rows['query_embedding'].iloc[0]))
    t_emb = ensure_2d(parse_embedding_field(rows['task_description_embedding'].iloc[0]))
    
    # Validate and prepare LLM embeddings
    llm_embeddings = np.asarray(llm_embeddings, dtype=float)
    if llm_embeddings.ndim == 1:
        raise ValueError("llm_embeddings is 1-D; expected (num_llms, dim)")
    if llm_embeddings.shape[0] not in (num_llms, 1):
        raise ValueError(
            f"llm_embeddings first dim ({llm_embeddings.shape[0]}) "
            f"must be {num_llms} or 1"
        )
    
    if llm_embeddings.shape[0] == 1 and num_llms > 1:
        llm_embeddings = np.tile(llm_embeddings, (num_llms, 1))
    
    # Extract effect and cost lists
    effect_list = np.array(rows['effect'].tolist(), dtype=float)
    cost_list = np.array(rows['cost'].tolist(), dtype=float)
    
    # Add semantic embeddings if available
    if semantic_embedding_path and os.path.exists(semantic_embedding_path):
        try:
            semantic_embeddings = load_pickle(semantic_embedding_path)
            semantic_embeddings = semantic_embeddings[query_id]
            semantic_embeddings = np.expand_dims(semantic_embeddings, axis=0)
            
            if semantic_embeddings.shape[0] == q_emb.shape[0]:
                q_emb = np.concatenate([q_emb, semantic_embeddings], axis=1)
                print(f"[INFO] Added semantic embeddings. New query feature dim = {q_emb.shape[1]}")
            else:
                print("[WARN] Semantic embeddings found but size mismatch, skipping.")
        except (KeyError, IndexError, FileNotFoundError) as e:
            print(f"[WARN] Could not load semantic embeddings: {e}")
    
    
    # Normalize effect and cost for scenario scoring
    effect_normalized = normalize_scores(effect_list)
    cost_normalized = normalize_scores(cost_list)
    # Compute scenario-based score
    score = compute_scenario_score(effect_normalized, cost_normalized, scenario)
    # print(score)
    

    # Build combined edge features
    edge_features = [cost_list.reshape(-1, 1), effect_list.reshape(-1, 1)]

    # ===================================================================
    # 🚀  Feedback Integration (real improvement happens here)
    # ===================================================================
    if 'avg_feedback' in df.columns:
        feedback_list = df['avg_feedback'].fillna(0)
        feedback_list = feedback_list.iloc[start:start + num_llms]
        feedback_list = np.array(feedback_list, dtype=float)
        
        # Normalize feedback if needed (assuming 1-5 scale)
        if feedback_list.max() > 1.0:
            feedback_list = (feedback_list - FEEDBACK_MIN_SCORE) / FEEDBACK_NORMALIZED_RANGE
        
        edge_features.append(feedback_list.reshape(-1, 1))

        # Normalize feedback
        feedback_n = (feedback_list - feedback_list.min()) / (feedback_list.max() - feedback_list.min() + 1e-12)
        # Impact of feedback (α: between 0.2..0.4 optimal)
        alpha = 0.40
        score = (1 - alpha) * score + alpha * feedback_n
        # print(score)
        # print("=== Impact of feedback added ===")


    combined_edge = np.concatenate(edge_features, axis=1)
    
    # Apply softmax with temperature scaling
    edge_feature = softmax(score / SOFTMAX_TEMPERATURE)
    # print(edge_feature)

    # Apply label smoothing
    label = (1 - LABEL_SMOOTHING) * edge_feature + (LABEL_SMOOTHING / len(edge_feature))
    # print(label)
    
    # Create node and mask arrays
    org_node = [0] * num_llms
    des_node = list(range(num_llms))
    mask_all = np.ones(num_llms, dtype=bool)
    
    # Build data dictionary
    data_dict = {
        "task_id": t_emb.astype(np.float32),
        "query_feature": q_emb.astype(np.float32),
        "llm_feature": llm_embeddings.astype(np.float32),
        "org_node": org_node,
        "des_node": des_node,
        "edge_feature": edge_feature.astype(np.float32),
        "label": label.astype(np.float32),
        "edge_mask": mask_all,
        "combined_edge": combined_edge.astype(np.float32),
        "train_mask": mask_all,
        "valide_mask": mask_all,
        "test_mask": mask_all
    }
    
    # Extract text descriptions
    query_text = (
        rows['query'].iloc[0]
        if 'query' in rows.columns
        else f"query_{query_id}"
    )
    task_text = (
        rows['task_description'].iloc[0]
        if 'task_description' in rows.columns
        else f"task_{query_id}"
    )
    
    return data_dict, (query_text, task_text), llm_names


# ============================================================================
# Inference Functions
# ============================================================================

def run_inference(
    query_id: int,
    saved_router_data_path: str,
    llm_embedding_path: str,
    llm_description_path: str,
    model_path: str,
    device: str,
    embedding_dim: int,
    edge_dim: int,
    scenario: str = DEFAULT_SCENARIO,
    checkpoint: Optional[str] = None,
    semantic_embedding_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run inference for a single query.
    
    Args:
        query_id: Query identifier
        saved_router_data_path: Path to router data CSV file
        llm_embedding_path: Path to LLM embeddings pickle file
        llm_description_path: Path to LLM descriptions JSON file
        model_path: Path to model checkpoint
        device: Device to run inference on ('cpu' or 'cuda')
        embedding_dim: Embedding dimension for the model
        edge_dim: Edge feature dimension
        scenario: Scenario name for scoring
        checkpoint: Optional path to specific checkpoint (overrides model_path)
        semantic_embedding_path: Optional path to semantic embeddings
        
    Returns:
        Dictionary containing:
            - query_id: Query identifier
            - query_text: Query text
            - task_text: Task description
            - best_llm: Name of best LLM
            - scores: Dictionary mapping LLM names to scores
            
    Raises:
        FileNotFoundError: If required files are missing
        RuntimeError: If model loading fails
    """
    # Load data
    df = pd.read_csv(saved_router_data_path)
    llm_embeddings = load_pickle(llm_embedding_path)
    
    # Build query data
    data_dict, (query_text, task_text), llm_names = build_single_query_data(
        df=df,
        llm_embeddings=llm_embeddings,
        query_id=query_id,
        scenario=scenario,
        llm_description_path=llm_description_path,
        semantic_embedding_path=semantic_embedding_path
    )
    
    # Formulate data for GNN
    form = form_data(device)
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
    
    # Convert masks to tensors
    edge_mask_t = to_bool_tensor(data_obj.edge_mask)
    edge_can_see = to_bool_tensor(data_obj.test_mask)
    
    # Get dimensions
    q_dim = data_obj.query_features.shape[1]
    llm_dim = data_obj.llm_features.shape[1]
    in_edges = edge_dim  # Use the same edge_dim as training
    
    # Initialize and load model
    model = EncoderDecoderNet(
        query_feature_dim=q_dim,
        llm_feature_dim=llm_dim,
        hidden_features=embedding_dim,
        in_edges=in_edges
    ).to(device)
    
    ckpt = checkpoint or model_path
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    
    # Run inference
    with torch.no_grad():
        pred = model(
            task_id=data_obj.task_id,
            query_features=data_obj.query_features,
            llm_features=data_obj.llm_features,
            edge_index=data_obj.edge_index,
            edge_mask=edge_mask_t,
            edge_can_see=edge_can_see,
            edge_weight=data_obj.combined_edge
        )
    
    pred = pred.reshape(-1, len(llm_names))
    
    # Get best LLM and scores
    best_idx = int(torch.argmax(pred, dim=1).cpu().item())
    scores = {
        llm_names[i]: float(pred[0, i].cpu().item())
        for i in range(len(llm_names))
    }
    
    return {
        "query_id": int(query_id),
        "query_text": query_text,
        "task_text": task_text,
        "best_llm": llm_names[best_idx],
        "scores": scores
    }


# ============================================================================
# Visualization Functions
# ============================================================================

def plot_llm_scores(
    query_id: int,
    saved_router_data_path: str,
    llm_embedding_path: str,
    llm_description_path: str,
    model_path: str,
    device: str,
    embedding_dim: int,
    edge_dim: int,
    output_dir: str,
    scenarios: Optional[List[str]] = None,
    plot: bool = True,
    semantic_embedding_path: Optional[str] = None
) -> pd.Series:
    """
    Run LLM inference for a given query and plot scores across scenarios.
    
    Args:
        query_id: The query index to run inference on
        saved_router_data_path: Path to router data CSV file
        llm_embedding_path: Path to LLM embeddings pickle file
        llm_description_path: Path to LLM descriptions JSON file
        model_path: Path to model checkpoint
        device: Device to run inference on
        embedding_dim: Embedding dimension for the model
        edge_dim: Edge feature dimension
        output_dir: Directory to save plots
        scenarios: List of scenario names (defaults to all scenarios)
        plot: Whether to create and save the plot
        semantic_embedding_path: Optional path to semantic embeddings
        
    Returns:
        Series containing LLM names from the Cost First scenario
    """
    if scenarios is None:
        scenarios = SCENARIOS
    
    results_list = []
    predicted_llms = []
    
    # Run inference for each scenario
    for i, scenario in enumerate(scenarios):
        out = run_inference(
            query_id=query_id,
            saved_router_data_path=saved_router_data_path,
            llm_embedding_path=llm_embedding_path,
            llm_description_path=llm_description_path,
            model_path=model_path,
            device=device,
            embedding_dim=embedding_dim,
            edge_dim=edge_dim,
            scenario=scenario,
            semantic_embedding_path=semantic_embedding_path
        )
        
        print()
        if i == 0:
            print(f"Query id: {out['query_id']} - Query: {out['query_text']}")
            print()
        
        print(f"-------------{scenario}-------------------")
        print(f"Best LLM: {out['best_llm']}")
        predicted_llms.append(out['best_llm'])
        
        for llm, score in out['scores'].items():
            results_list.append({
                "LLM": llm,
                "Score": score,
                "Scenario": scenario
            })
    
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
        
        # Save as HTML file
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"llm_scores_query_{query_id}.html")
        fig.write_html(output_file)
        print(f"Graph saved as: {output_file}")
        
        pio.renderers.default = "browser"
        fig.show()
    
    # Return LLM names from Cost First scenario
    # cost_first_scores = df_scores[df_scores["Scenario"] == "Cost First"]
    # print(cost_first_scores['LLM'])
    return predicted_llms


# ============================================================================
# Feedback Functions
# ============================================================================

def save_feedback(
    query_id: int,
    llm_name: str,
    user_score: float,
    feedback_path: str,
    extra: Optional[Dict[str, Any]] = None
) -> None:
    """
    Save user feedback to a file.
    
    Args:
        query_id: Query identifier
        llm_name: Name of the LLM
        user_score: User-provided score
        feedback_path: Path to feedback file
        extra: Optional additional metadata
    """
    record = {
        "timestamp": time.time(),
        "query_id": int(query_id),
        "LLM": llm_name,
        "Score": float(user_score),
    }
    
    if extra:
        record.update(extra)
    
    os.makedirs(os.path.dirname(feedback_path), exist_ok=True)
    with open(feedback_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def ask_and_save_feedback(
    query_id: int,
    predicted_llm: str,
    feedback_path: str
) -> None:
    """
    Prompt user for feedback and save it.
    
    Args:
        query_id: Query identifier
        predicted_llm: Name of predicted LLM
        feedback_path: Path to feedback file
    """
    user_input = input("\n===========> Score 1..5 (or blank to reject): ").strip()
    if user_input.isdigit():
        score = int(user_input)
        save_feedback(query_id, predicted_llm, score, feedback_path)


def auto_score_feedback(
    query_id: int,
    predicted_llms: pd.Series,
    feedback_path: str,
    penalized_llm: str = "NousResearch"
) -> None:
    """
    Automatically assign feedback scores with penalty for specific LLM.
    
    Args:
        query_id: Query identifier
        predicted_llms: Series of predicted LLM names
        feedback_path: Path to feedback file
        penalized_llm: LLM name to penalize (lower scores)
    """
    penalized_llm_ = [l.lower() for l in penalized_llm] 

    for predicted_llm in predicted_llms:
        llm_name_lower = predicted_llm.strip().lower()
        
        if llm_name_lower in penalized_llm_:
            # score = random.randint(1, 2)  # Penalize
            score = 1
        else:
            # score = random.randint(3, 5)  # Normal range
            score = 4
        
        save_feedback(
            query_id,
            predicted_llm,
            score,
            feedback_path,
            extra={"auto": True}
        )
        print(f"[Feedback Saved] Query {query_id} | {predicted_llm} → Score: {score}")


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main execution function."""
    # Configuration
    device = "cpu"
    # model_path = 'model_path/best_model_nlgFA_Cost_First+semantic_embedding/best_f1.pt'
    model_path = 'model_path/best_model_MIZAN_PersianNLG_CostFirst+feedback+semantic_embeddings/best_f1.pt'

    
    embedding_dim = 8
    edge_dim = 3

    data_dir = "data/MIZAN_PersianNLG"
    saved_router_data_path = os.path.join(data_dir, 'router_data.csv')
    llm_embedding_path = os.path.join(data_dir, "llm_description_embedding.pkl")
    llm_description_path = os.path.join(data_dir, 'LLM_Descriptions.json')
    
    semantic_embedding_path = None
    if 'semantic' in model_path:
        semantic_embedding_path = os.path.join(data_dir, "query_semantic_embeddings.pkl")
    if 'feedback' in model_path:
        saved_router_data_path = os.path.join(data_dir, 'feedback/router_data.csv')
        edge_dim = 4


    output_dir = os.path.join("inference_results", model_path.split('/')[1])
    os.makedirs(output_dir, exist_ok=True)
    
    feedback_path = os.path.join(data_dir, "feedback", "feedback.jsonl")
    os.makedirs(os.path.dirname(feedback_path), exist_ok=True)
    

    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run LLM inference for queries")
    parser.add_argument("--query_id", type=int, default=0, help="Query ID to run inference on")

    parser.add_argument(
        "--generate_feedback",
        action="store_true",
        help="Generate automatic feedback for all queries"
    )
    parser.add_argument(
        "--penalized_llm",
        type=list,
        default=["Gemma-3N (e4b)", "gpt-oss-20B"],
        help="LLM name to penalize in auto feedback"
    )
    args = parser.parse_args()
    
    if not args.generate_feedback:
        # Run inference for single query
        predicted_llm = plot_llm_scores(
            query_id=args.query_id,
            saved_router_data_path=saved_router_data_path,
            llm_embedding_path=llm_embedding_path,
            llm_description_path=llm_description_path,
            model_path=model_path,
            device=device,
            embedding_dim=embedding_dim,
            edge_dim=edge_dim,
            output_dir=output_dir,
            semantic_embedding_path=semantic_embedding_path
        )
        
    # Optionally ask for manual feedback
    # ask_and_save_feedback(args.query_id, predicted_llm.iloc[0], feedback_path)
    
    # ====================================================================== #

    # Generate feedback for all queries if requested
    if args.generate_feedback:
        df = pd.read_csv(saved_router_data_path)
        llm_desc = load_json(llm_description_path)
        llm_names = list(llm_desc.keys())
        num_llms = len(llm_names)
        num_queries = int(len(df) / num_llms)
        
        print(f"DataFrame size: {len(df)}")
        print(f"Number of LLMs: {num_llms}")
        print(f"Running inference for {num_queries} queries...")
        
        for qid in range(num_queries):
            # try:
            print(f"Processing query {qid}...")
            predicted_llms = plot_llm_scores(
                query_id=qid,
                saved_router_data_path=saved_router_data_path,
                llm_embedding_path=llm_embedding_path,
                llm_description_path=llm_description_path,
                model_path=model_path,
                device=device,
                embedding_dim=embedding_dim,
                edge_dim=edge_dim,
                output_dir=output_dir,
                plot=False,
                semantic_embedding_path=semantic_embedding_path
            )
            print(f"Predicted LLMs: {predicted_llms}")
            auto_score_feedback(qid, predicted_llms, feedback_path, args.penalized_llm)
            # except Exception as e:
                # print(f"⚠️ Skipped query {qid} due to error: {e}")


if __name__ == "__main__":
    main()
