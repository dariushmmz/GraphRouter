"""
Tricks for Adapting GraphRouter to Other Tasks and Datasets
   * Embedding Normalization
        - Check whether input embeddings are normalized.
        - On some datasets, skipping normalization leads to suboptimal results.
"""

import pandas as pd
import numpy as np
import pickle


import os
import yaml

with open("data/config.yaml", 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)

data_dir = config['data_dir']
ROUTER_DATA_PATH = os.path.join(data_dir, 'router_data.csv')
LLM_EMBEDDING_PATH = os.path.join(data_dir.replace('data', 'configs'), 'llm_description_embedding.pkl')
SEMANTIC_EMBEDDING_PATH = os.path.join(data_dir, 'query_semantic_embeddings.pkl')

if config['feedback']:
    router_data_path = os.path.join(data_dir, 'feedback/router_data.csv')
    if not os.path.exists(router_data_path):
        print("[INFO] No feedback found")
        router_data_path = os.path.join(data_dir, 'router_data.csv')
        

# ---------------------------
# 1. Load CSV & PKL
# ---------------------------
df = pd.read_csv(ROUTER_DATA_PATH)

# ---------------------------
# Safe parser for numpy-like embeddings
# ---------------------------
def parse_embedding(emb):
    if isinstance(emb, str):
        emb = emb.replace("\n", " ")  # remove line breaks
        return np.fromstring(emb.strip("[]"), sep=" ")
    return np.array(emb, dtype=np.float32)


# ---------------------------
# Normalize vector
# ---------------------------
def normalize(vec):
    vec = np.array(vec, dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm

# ---------------------------
# Convert vector BACK to original format
# ---------------------------
def to_original_format(vec):
    return np.array2string(
        vec,
        separator=' ',        # <-- no commas, like original
        max_line_width=10**6, # prevent line wrapping
    )


# ---------------------------
# Apply normalization to df
# ---------------------------
def normalize_df_embeddings(row, col):
    emb = parse_embedding(row[col])
    emb = normalize(emb)
    return to_original_format(emb)

df["query_embedding"] = df.apply(
    normalize_df_embeddings, axis=1, args=("query_embedding",)
)

df["task_description_embedding"] = df.apply(
    normalize_df_embeddings, axis=1, args=("task_description_embedding",)
)

# ---------------------------
# Save results
# ---------------------------
# ROUTER_DATA_PATH_NORMALIZED = ROUTER_DATA_PATH.replace(".csv", "_normalized.csv")
df.to_csv(ROUTER_DATA_PATH, index=False)


# ---------------------------
# Load embeddings
# ---------------------------
with open(LLM_EMBEDDING_PATH, "rb") as f:
    description_embeddings = pickle.load(f)

with open(SEMANTIC_EMBEDDING_PATH, "rb") as f:
    semantic_embeddings = pickle.load(f)


# ---------------------------
# Normalize embeddings
# ---------------------------
if isinstance(description_embeddings, dict):
    # If embeddings are stored as a dict
    for key in description_embeddings:
        description_embeddings[key] = normalize(description_embeddings[key])

    for key in semantic_embeddings:
        semantic_embeddings[key] = normalize(semantic_embeddings[key])
else:
    # If embeddings are stored as a 2D array
    description_embeddings = np.array([normalize(v) for v in description_embeddings], dtype=np.float32)
    semantic_embeddings = np.array([normalize(v) for v in semantic_embeddings], dtype=np.float32)



# ---------------------------
# Save back as 2D array
# ---------------------------
# LLM_EMBEDDING_PATH_NORMALIZED = LLM_EMBEDDING_PATH.replace(".pkl", "_normalized.pkl")
with open(LLM_EMBEDDING_PATH, "wb") as f:
    pickle.dump(description_embeddings, f)

# SEMANTIC_EMBEDDING_PATH_NORMALIZED = SEMANTIC_EMBEDDING_PATH.replace(".pkl", "_normalized.pkl")
with open(SEMANTIC_EMBEDDING_PATH, "wb") as f:
    pickle.dump(semantic_embeddings, f)


print("✓ Normalization complete!")
print("Format preserved exactly like original.")