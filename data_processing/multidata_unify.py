import os
import yaml
import pandas as pd
from typing import Dict, List, Optional, Union
import re

from utils import loadjson
from instructions import *

# ------------------------------------------------------------------
# Dataset Configs
# ------------------------------------------------------------------

DATASET_CONFIGS: List[Dict] = [
    {
        "task_name": "alpaca_data",
        "path": "data/alpaca_data/alpaca_data.json",
        "format": "json",
        "query_fields": ["instruction", "input"],
        "ground_truth_field": "output",
        "metric": "f1_score",
        "task_description": "Instruction-following dataset for general-purpose QA."
    },
    {
        "task_name": "GSM8K",
        "path": [
            "data/GSM8K/train-00000-of-00001.parquet",
            "data/GSM8K/test-00000-of-00001.parquet",
        ],
        "format": "parquet",
        "query_fields": ["question"],
        "ground_truth_field": "answer",
        "metric": "exact_match",
        "task_description": "The GSM8K dataset is tailored for mathematical problem-solving tasks. It consists of natural language math problems that require the model to comprehend the problem statement, apply the correct mathematical operations, and provide the solution. The primary challenge lies in both parsing complex language and performing accurate calculations."
        
    },
    {
        "task_name": "hendrycks-MATH",
        "path": [
            "data/hendrycks-MATH/train-00000-of-00001.parquet",
            "data/hendrycks-MATH/test-00000-of-00001.parquet",
        ],
        "format": "parquet",
        "query_fields": ["problem"],
        "ground_truth_fields": ["solution", "answer"],
        "metric": "exact_match",
        "task_description": "The hendrycks-MATH-benchmark (based on the MATH dataset) is designed for advanced mathematical reasoning and problem-solving tasks. It consists of high school competition-level mathematics problems (drawn from sources like AMC 10/12, AIME) across subjects such as Prealgebra, Algebra, Number Theory, Counting & Probability, Geometry, Intermediate Algebra, and Precalculus, with difficulty levels from 1 (easiest) to 5 (hardest). The model must comprehend complex problem statements, apply appropriate mathematical concepts and techniques, perform step-by-step derivations or proofs when needed, and arrive at the correct final numerical or symbolic answer (often presented in boxed format). The primary challenge lies in deep conceptual understanding, multi-step logical reasoning, and accurate computation without relying on simple pattern matching."
    },
    {
        "task_name": "multi_news",
        "path": "data/multi_news/multi_news.json",
        "format": "json",
        "query_fields": ["instruction", "input"],
        "ground_truth_field": "output",
        "metric": "f1_score",
        "task_description": "Multi-document summarization."
    },
    {
        "task_name": "SQUAD",
        "path": "data/SQUAD/SQUAD.parquet",
        "format": "parquet",
        "query_fields": ["question"],
        "ground_truth_field": "answers",
        "ground_truth_subfield": "text",
        "ground_truth_index": 0,
        "metric": "f1_score",
        "task_description": "Extractive question answering."
    },
]


# ------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------

def load_parquet_files(paths: Union[str, List[str]]) -> pd.DataFrame:
    """Load one or multiple parquet files into a single DataFrame."""
    if isinstance(paths, str):
        paths = [paths]

    frames = [pd.read_parquet(p) for p in paths]
    return pd.concat(frames, ignore_index=True)


def build_query(row: pd.Series, fields: List[str]) -> str:
    """Construct query text from one or more fields."""
    return " ".join(str(row[f]) for f in fields if f in row)


# ------------------------------------------------------------------
# Main Generator
# ------------------------------------------------------------------

def generate_unified_qa_dataset(
    output_dir: str,
    task_name: str,
    sample_size: Optional[int] = None
) -> pd.DataFrame:
    """
    Generate a unified QA dataset for a single task.

    Parameters
    ----------
    output_dir : str
        Directory to save unified CSV.
    task_name : str
        Dataset task name.
    sample_size : Optional[int]
        Limit number of samples.

    Returns
    -------
    pd.DataFrame
    """

    config_map = {cfg["task_name"]: cfg for cfg in DATASET_CONFIGS}

    if task_name not in config_map:
        raise ValueError(f"Unknown task_name: {task_name}")

    config = config_map[task_name]
    rows = []

    # -----------------------------
    # JSON DATASETS
    # -----------------------------
    if config["format"] == "json":
        data = loadjson(config["path"])
        if sample_size:
            data = data[:sample_size]

        for item in data:
            query = " ".join(str(item[f]) for f in config["query_fields"])
            ground_truth = item[config["ground_truth_field"]]

            rows.append({
                "task_id": config["task_name"],
                "query": query,
                "ground_truth": ground_truth,
                "metric": config["metric"],
                "task_description": config["task_description"],
            })

    # -----------------------------
    # PARQUET DATASETS (STANDARDIZED)
    # -----------------------------
    elif config["format"] == "parquet":
        df = load_parquet_files(config["path"])
        if sample_size:
            df = df.head(sample_size)

        for _, row in df.iterrows():
            query = build_query(row, config["query_fields"])

            # Hendrycks-MATH special handling
            if config["task_name"] == "hendrycks-MATH":
                query = f"{MATH_INSTRUCTION}\n\nQuestion:\n{query}"
                solution, answer = config["ground_truth_fields"]
                # فقط diagramهای [asy] ... [/asy] را حذف کن (برای embedding و semantic similarity نویز هستند)

                solution_text = row[solution]
                # Regular expression to remove all [asy]...[/asy] blocks (non-greedy, possibly multiline)
                solution_text = re.sub(r"\[asy\].*?\[/asy\]", "", solution_text, flags=re.DOTALL).strip()
                ground_truth = f"{solution_text}\n\n####\n\n{row[answer].strip()}"
            elif config['task_name'] == "GSM8K":
                query = f"{MATH_GSM8K_INSTRUCTION}\n{query}"
                ground_truth = row[config["ground_truth_field"]]
                


            # SQuAD handling
            else:
                answers = row[config["ground_truth_field"]]
                ground_truth = answers[config["ground_truth_index"]][
                    config["ground_truth_subfield"]
                ]

            rows.append({
                "task_id": config["task_name"],
                "query": query,
                "ground_truth": ground_truth,
                "metric": config["metric"],
                "task_description": config["task_description"],
            })

    else:
        raise ValueError(f"Unsupported format: {config['format']}")

    df_out = pd.DataFrame(rows)
    os.makedirs(output_dir, exist_ok=True)
    df_out.to_csv(os.path.join(output_dir, "unified_data.csv"), index=False)

    return df_out


# ------------------------------------------------------------------
# Entry Point
# ------------------------------------------------------------------

if __name__ == "__main__":
    with open("configs/config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    generate_unified_qa_dataset(
        output_dir=cfg["data_dir"],
        task_name=cfg["data_dir"].split('/')[-1],
        sample_size=cfg.get("sample_size", None),
    )
