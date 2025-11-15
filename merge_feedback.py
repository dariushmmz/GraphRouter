import json
import pandas as pd
from collections import defaultdict
import numpy as np
import os


def merge_feedback(router_csv="data/router_data.csv",
                   feedback_jsonl="data/feedback.jsonl",
                   out_csv="data/router_data_with_feedback.csv"):
    
    df = pd.read_csv(router_csv)

    # detect LLM name column
    llm_col = None
    for c in ["llm", "LLM", "llm_name"]:
        if c in df.columns:
            llm_col = c
            break
    if llm_col is None:
        raise ValueError("No 'llm' column found in router_data.csv")

    # load feedback
    scores = defaultdict(list)

    if not os.path.exists(feedback_jsonl):
        print("No feedback file; writing empty feedback column.")
        df["avg_feedback"] = np.nan
        df.to_csv(out_csv, index=False)
        return

    with open(feedback_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            qid = int(rec["query_id"])
            llm = str(rec["LLM"]).strip()
            scores[(qid, llm)].append(float(rec["Score"]))

    # load llm order from config
    llm_desc_path = "configs/LLM_Descriptions.json"
    with open(llm_desc_path, "r", encoding="utf-8") as f:
        llm_names = list(json.load(f).keys())

    num_llms = len(llm_names)
    total_rows = len(df)

    if total_rows % num_llms != 0:
        raise ValueError("Rows not divisible by num_llms!")

    # ---- compute avg feedback per row ----
    avg_list = []

    for idx, row in df.iterrows():
        query_id = idx // num_llms        # correct query index
        llm_name = str(row[llm_col]).strip()
        key = (query_id, llm_name)

        if key in scores:
            avg_list.append(np.mean(scores[key]))
        else:
            avg_list.append(np.nan)

    df["avg_feedback"] = avg_list
    print(avg_list)

    df.to_csv(out_csv, index=False)
    print(f"Saved merged feedback → {out_csv}")

merge_feedback()