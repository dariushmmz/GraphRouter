import json, numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union


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

# Softmax temperature for score sharpening
SOFTMAX_TEMPERATURE = 0.15
LABEL_SMOOTHING = 0.05


# Feedback score normalization
FEEDBACK_MAX_SCORE = 5.0
FEEDBACK_MIN_SCORE = 1.0
FEEDBACK_NORMALIZED_RANGE = 4.0

# ============================================================================
# Adaptive Batch-Delayed Reward System
# ============================================================================
import os
from dotenv import load_dotenv

load_dotenv()
os.environ["TOGETHERAI_API_KEY"] = os.getenv("api_key")

def get_responsne(query, llm_name, query_id, llm_description):
    from data_processing.llm_engine import LLMEngine
    engine = LLMEngine(
        llm_description=llm_description
        )

    try:
        response = engine.get_llm_response(
            query=query,
            llm_name=llm_name,
        )
        
        items = response.split('A:')
        if not len(items)>1:
            print(f">>>>>> No structred output !! query_id: {query_id}")

        return response

    except Exception as e:
        raise f"LLM failed: {e}"


class AdaptiveBatchUpdater:
    """
    Professional Batch-Delayed Adaptive Edge Weight Updater.
    Supports:
        - OMS-based reward
        - Feedback-based reward
        - Combined reward
    """

    def __init__(
        self,
        router_csv_path: str,
        llm_description_path: str,
        batch_size: int = 40,
        eta: float = 0.05,
        use_feedback: bool = False,
        beta: float = 0.3
    ):
        self.router_csv_path = router_csv_path
        self.batch_size = batch_size
        self.eta = eta
        self.use_feedback = use_feedback
        self.beta = beta

        self.df = pd.read_csv(router_csv_path)

        self.llm_desc = load_json(llm_description_path)
        self.llm_names = list(self.llm_desc.keys())
        self.num_llms = len(self.llm_names)

        self.reward_buffer = []

    # --------------------------------------------------
    # Register reward (OMS or combined)
    # --------------------------------------------------

    def register_reward(
        self,
        query_id: int,
        llm_name: str,
        oms_value: float,
        feedback_score: Optional[float] = None
    ):
        if llm_name not in self.llm_names:
            return

        model_id = self.llm_names.index(llm_name)

        reward = float(oms_value)

        # Combine with feedback if enabled
        if self.use_feedback and feedback_score is not None:
            feedback_norm = (
                (feedback_score - FEEDBACK_MIN_SCORE)
                / FEEDBACK_NORMALIZED_RANGE
            )
            reward = (1 - self.beta) * reward + self.beta * feedback_norm

        self.reward_buffer.append((query_id, model_id, reward))

        if len(self.reward_buffer) >= self.batch_size:
            self.batch_update()

    # --------------------------------------------------
    # Batch update
    # --------------------------------------------------

    def batch_update(self):

        if not self.reward_buffer:
            return

        print(f"[Adaptive Update] Processing {len(self.reward_buffer)} samples")

        reward_dict = {}

        for query_id, model_id, reward in self.reward_buffer:
            reward_dict.setdefault((query_id, model_id), []).append(reward)

        for (query_id, model_id), rewards in reward_dict.items():

            mean_reward = np.mean(rewards)

            start = query_id * self.num_llms
            edge_index = start + model_id

            old_effect = float(self.df.loc[edge_index, "effect"])

            # EMA Update (Delayed Learning)
            new_effect = old_effect + self.eta * (mean_reward - old_effect)

            self.df.loc[edge_index, "effect"] = new_effect


        # Save persistent router state
        self.df.to_csv(self.router_csv_path, index=False)
        print("[Adaptive Update] Router weights saved.")

        self.reward_buffer = []

    # --------------------------------------------------
    # Final flush
    # --------------------------------------------------

    def flush(self):
        self.batch_update()


    def compute(self, query_text, feedback_score, llm, query_id, num_llm):
        llm_response = get_responsne(query_text, llm, query_id, self.llm_desc)
        if llm_response:
            from data_processing.oms_metric import extract_reasoning, get_embedding, compute_rqs, compute_oms
            row = self.df.iloc[query_id*num_llm]

            model_reasoning = extract_reasoning(llm_response)
            if model_reasoning == "":
                return 0

            gt_reasoning = row['ground_truth_reasoning']

            model_emb = get_embedding([model_reasoning])[0]
            gt_emb = get_embedding([gt_reasoning])[0]
            alpha=0.5
            beta=0.3

            rqs = compute_rqs(
                model_reasoning,
                model_emb,
                gt_emb,
                alpha=alpha
            )


            reward = compute_oms(row['EM_metric'], rqs, beta)

            self.register_reward(
                query_id=query_id,
                llm_name=llm,
                oms_value=reward,
                feedback_score=feedback_score
            )

