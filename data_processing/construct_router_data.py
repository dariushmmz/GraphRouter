import os
import re
import yaml
import time
import pandas as pd
from transformers import AutoTokenizer

from llm_engine import LLMEngine
from utils import *
from instructions import *


class DataBuilderIncremental:

    def __init__(self, unified_path, llm_path, config, llm_names: list= []):
        log("Initializing DataBuilderIncremental")

        self.unified_data = pd.read_csv(unified_path)
        self.llm_description = loadjson(llm_path)
        self.llm_names = llm_names
        if not len(llm_names)>=1:
            self.llm_names = list(self.llm_description.keys())
        
        self.all_llm_description = [
            self.llm_description[n]['feature'] for n in self.llm_names
        ]

        self.engine = LLMEngine(
            llm_description=self.llm_description
        )

        self.config = config
        self.output_file = os.path.join(
            self.config['data_dir'], "router_data.csv"
        )

        self._load_existing()
        self._run()


    # --------------------------------------------------
    # Resume support
    # --------------------------------------------------
    def _load_existing(self):
        if os.path.exists(self.output_file):
            df_existing = pd.read_csv(self.output_file)
            if 'OMS' in df_existing.columns:
                df_existing = df_existing.drop('OMS', axis=1)
            self.processed_uids = set(df_existing['uid'])
            self.df_existing = df_existing
            log(f"Resuming from existing file ({len(self.processed_uids)} rows)")
        else:
            self.processed_uids = set()
            log("Starting from scratch")


    # --------------------------------------------------
    # Main loop
    # --------------------------------------------------
    def _run(self):
        max_samples = self.config.get('max_samples', None)
        samples_processed = 0

        log(f"Total unified_data rows: {len(self.unified_data)}")

        if max_samples is not None:
            log(f"Max samples limit: {max_samples}")

        for row_idx, row in enumerate(self.unified_data.itertuples(), start=1):
            if max_samples is not None and samples_processed >= max_samples:
                log("Reached max_samples limit. Stopping.")
                break

            task_id = row.task_id
            query = row.query
            query = query.replace('[asy]', '').replace('[/asy]', '')
            task_description = row.task_description
            metric = row.metric

            ########################### temp            
            import re
            def normalize(t):
                return re.sub(r'\s+', ' ', t).strip().lower()
            result = self.df_existing[
                self.df_existing['query'].apply(normalize) == normalize(query)
            ]
            if len(result)>=1:
                log(f"SKIP | query already processed")  
                print()
                continue

            ################################

            print('\n\n')
            log(f"Sample {row_idx} | query_id={stable_hash(query)} | metric={metric}")
            # log(f"Query preview: {query[:120]}{'...' if len(query) > 120 else ''}")

            if task_id == "multi_news":
                tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
                original_len = len(tokenizer.tokenize(query))
                tokens = tokenizer.tokenize(query)[:3000]
                query = tokenizer.convert_tokens_to_string(tokens)
                log(f"Multi-News truncation: {original_len} → {len(tokens)} tokens")

            log("Computing embeddings")
            if task_id == 'GSM8K':
                query_embedding = get_embedding([query.replace(MATH_GSM8K_INSTRUCTION, '').replace('\n', '')])
            else:
                query_embedding = get_embedding([query.replace(MATH_INSTRUCTION, '').replace('\n', '')])

            task_desc_embedding = get_embedding([task_description])

            ground_truth_final = extract_final_answer(row.ground_truth)
            ground_truth_reasoning = extract_reasoning(row.ground_truth)

            # if not ground_truth_reasoning

            log(f"Ground truth final: {ground_truth_final}")
            print("========================================>")

            sample_has_unprocessed = False

            for llm_idx, llm_name in enumerate(self.llm_names):

                uid, uid_query = build_uid(task_id, query, llm_name)

                if uid in self.processed_uids:
                    log(f"SKIP | {llm_name} already processed in {uid_query}")
                    print()

                    continue

            

                sample_has_unprocessed = True
                log(f"CALL | LLM={llm_name} (idx={llm_idx})")

                try:
                    response = self.engine.get_llm_response(
                        query=query,
                        llm_name=llm_name,
                        max_token=self.config.get("max_token", 512)
                    )
                    
                    items = response.split('A:')
                    if not len(items)>1:
                        log(f">>>>>> No structred output !! query_uid: {uid_query}")

                except Exception as e:
                    log(f"LLM failed: {e}", level="ERROR")
                    print()
                    
                    time.sleep(5)
                    continue

                # log(f"Response preview: {response[:150]}{'...' if len(response) > 150 else ''}")

                # ---------------- Evaluation ----------------
                reward = 0

                if metric in {"exact_match", "GSM8K"}:
                    pred = extract_final_answer_from_response(response)
                    # log(f"Predicted answer: {pred[:10]}")

                    if pred is not None and ground_truth_final is not None:
                        if is_numeric_ground_truth(ground_truth_final):
                            pred_num = normalize_numeric_answer(pred)
                            if pred_num is not None:
                                log(f"Numeric compare | pred={pred_num[:10]} | gt={ground_truth_final[:10]}")
                                reward = answer_match(pred_num, ground_truth_final)
                            else: 
                                reward = 0
                            # reward = int(str(pred_num) == str(ground_truth_final))
                        else:
                            log(f"String compare | pred={pred[:10]} | gt={ground_truth_final[:10]}")
                            # reward = int(str(pred) == str(ground_truth_final))
                            reward = answer_match(pred, ground_truth_final)

                else:
                    reward = self.engine.eval(
                        prediction=response,
                        ground_truth=row.ground_truth,
                        metric=metric
                    )

                log(f"Reward: {reward}")

                cost = self.engine.compute_cost(
                    llm_name=llm_name,
                    input_text=query,
                    output_size=512
                )

                log(f"Cost: {cost}")

                record = {
                    "uid": uid,
                    "task_id": task_id,
                    "task_description": task_description,
                    "task_description_embedding": task_desc_embedding,
                    "query": query,
                    "query_embedding": query_embedding,
                    "ground_truth_numeric": ground_truth_final,
                    "ground_truth_reasoning": ground_truth_reasoning,
                    "model_response": response,
                    "metric": metric,
                    "llm": llm_name,
                    "effect": reward,
                    "cost": cost,
                    "query_uid": uid_query
                }

                self._append_row(record)
                self.processed_uids.add(uid)

                log(f"SAVED | uid={uid} | query uid={uid_query}")
                log("__________________________________________")

                sleep_time = self.config.get("sleep_between_calls", 0)
                if sleep_time > 0:
                    log(f"Sleeping {sleep_time}s")
                    time.sleep(sleep_time)

            if sample_has_unprocessed:
                samples_processed += 1
                log(f"Samples processed: {samples_processed}")

            print()
        self._postprocess()


    # --------------------------------------------------
    # CSV append
    # --------------------------------------------------
    def _append_row(self, record):
        if record["uid"] in self.processed_uids:
            log(f"DUPLICATE PREVENTED | uid={record['uid']}")
            return

        df = pd.DataFrame([record])
        df.to_csv(
            self.output_file,
            mode='a',
            header=not os.path.exists(self.output_file),
            index=False
        )


    # --------------------------------------------------
    # Post processing
    # --------------------------------------------------
    def _postprocess(self):
        log("Starting post-processing")

        df = pd.read_csv(self.output_file)
        log(f"Rows loaded: {len(df)}")

        df['cost'] = df.groupby('task_id')['cost'].transform(
            lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8)
        )

        df.to_csv(self.output_file, index=False)
        log("Cost normalization done")

        llm_desc_emb = get_embedding(self.all_llm_description)
        savepkl(
            llm_desc_emb,
            os.path.join(self.config['data_dir'].replace('data', 'configs'), "llm_description_embedding.pkl")
        )

        log("LLM description embeddings saved")
        log("Dataset construction completed", level="DONE")


# ======================================================
# Entry point
# ======================================================

if __name__ == "__main__":

    from dotenv import load_dotenv
    load_dotenv()

    with open("configs/config.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    os.environ["TOGETHERAI_API_KEY"] = os.getenv("api_key")
    os.environ["OPENROUTER_API_KEY"] = os.getenv("openrouter_api_key")
# 
    data_dir = config['data_dir']

    DataBuilderIncremental(
        unified_path=os.path.join(data_dir, "unified_data.csv"),
        llm_path=os.path.join(data_dir, "LLM_Descriptions.json"),
        config=config,
        # llm_names=['Gemini 2.5 Pro']
    )
