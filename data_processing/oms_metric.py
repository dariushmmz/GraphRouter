from utils import get_embedding
import re

def extract_reasoning(response_text):
    if response_text is None:
        return ""
    if "A:" in response_text:
        return response_text.split("A:")[0]
    return response_text


def reasoning_structure_score(reasoning_text):
    if not reasoning_text:
        return 0.0

    score = 0

    # step indicators
    if re.search(r"(first|then|next|finally|step)", reasoning_text, re.I):
        score += 0.4

    # math operators
    if re.search(r"[\+\-\*/=]", reasoning_text):
        score += 0.3

    # length (not too short)
    if len(reasoning_text.split()) > 20:
        score += 0.3

    return min(score, 1.0)

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# def semantic_similarity(emb1, emb2):
#     emb1 = np.array(emb1).reshape(1, -1)
#     emb2 = np.array(emb2).reshape(1, -1)
#     return cosine_similarity(emb1, emb2)[0][0]

def semantic_dot(query_emb, candidate_embs):
    """
    query_emb: shape (d,)
    candidate_embs: shape (N, d)
    """
    query_emb = np.array(query_emb)
    candidate_embs = np.array(candidate_embs)

    # dot product
    scores = np.dot(candidate_embs, query_emb)


    return scores

def compute_rqs(model_reasoning,
                model_emb,
                gt_emb,
                alpha=0.5):

    structure = reasoning_structure_score(model_reasoning)
    semantic = semantic_dot(model_emb, gt_emb)

    return alpha * structure + (1 - alpha) * semantic

def compute_oms(em, rs, gamma=0.5):
    """
    em: 0 or 1 (exact match)
    rs: float in [0,1] (reasoning score)
    gamma: weight on exact match (same as γ in the formula)
    """
    return gamma * em + (1 - gamma) * rs


def add_oms_to_dataframe(df, alpha=0.5, beta=0.3):

    oms_list = []
    rqs_list = []


    for _, row in df.iterrows():
        model_reasoning = extract_reasoning(row['model_response'])
        model_reasoning = model_reasoning.replace('R:', '')

        gt_reasoning = row['ground_truth_reasoning']

        model_emb = get_embedding([model_reasoning])[0]
        gt_emb = get_embedding([gt_reasoning])[0]

        rqs = compute_rqs(
            model_reasoning,
            model_emb,
            gt_emb,
            alpha=alpha
        )

        oms = compute_oms(row['RM_metric'], rqs, beta)
        oms_list.append(oms)
        rqs_list.append(rqs)

    df['EM_metric'] = df['RM_metric']
    df = df.drop(['RM_metric', 'OMS'], axis=1)
    df['RQS_metric'] = rqs_list
    df['effect'] = oms_list
    return df


if __name__ == "__main__":
    import os
    import yaml, pandas as pd 

    os.environ["KMP_DUPLICATE_LIB_OK"] = 'True'
    # with open("configs/config.yaml", 'r', encoding='utf-8') as file:
        # config = yaml.safe_load(file)
    # os.environ["TOGETHERAI_API_KEY"] = config["api_key"]
    # data_dir = config['data_dir']

    data_dir = [
        # {
        # 'path': 'data/GSM8k',
        # 'difficulty': 'Easy'
        # },
        {
        'path': 'data/hendrycks-MATH',
        'difficulty': 'Hard'
        }
    ]


    for item in data_dir:
        path = item['path']
        difficulty = item['difficulty']

        router_path = os.path.join(path, "_router_data.csv")
        df = pd.read_csv(router_path)
        oms_df = add_oms_to_dataframe(df)
        oms_df.to_csv(router_path, index=False)
        


        summary = (
        df.groupby(['task_id', 'llm'])
        .agg(
            Cost=('cost', 'mean'),
            CR=('EM_metric', 'mean'),
            OMS=('effect', 'mean'),
        )
        .reset_index()
        )

        summary['Normalized_CR'] = summary.groupby('task_id')['CR'].transform(
            lambda x: (x - x.min()) / (x.max() - x.min())
        )

        summary['Normalized_OMS'] = summary.groupby('task_id')['OMS'].transform(
            lambda x: (x - x.min()) / (x.max() - x.min())
        )

        
        # Add 'difficulty' column to the summary dataframe for each row based on dataset
        summary.insert(1, 'difficulty', difficulty)
      # rename columns
        summary = summary.rename(columns={
            'task_id': 'data',
        })


        summary.to_csv(os.path.join(path, "final_data.csv"), index=False)
