import re
import os
from llm_engine import LLMEngine
from utils import savepkl, loadjson, get_embedding
import pandas as pd
import yaml
from transformers import AutoTokenizer


def extract_gsm8k_final_answer(answer_text):
    if "####" in answer_text:
        return answer_text.split("####")[-1].strip()
    return None


def extract_gsm8k_reasoning(answer_text):
    if "####" in answer_text:
        return answer_text.split("####")[0].strip()
    return answer_text


import re

def extract_final_answer_from_response(response_text):
    """
    Extract final numeric answer ONLY from the Answer (A:) section.
    """

    if response_text is None:
        return None

    # Normalize text
    text = response_text.replace("\n", " ").strip()

    # --------------------------------------------------
    # 1️⃣ Extract Answer section (A:)
    # --------------------------------------------------
    answer_part = None

    # Common patterns for Answer section
    answer_patterns = [
        r"A:\s*(.*)$",        # A: ....
        r"Answer:\s*(.*)$"    # Answer: ....
    ]

    for pattern in answer_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            answer_part = match.group(1).strip()
            break

    # If no explicit Answer section found, fallback (last sentence)
    if answer_part is None:
        answer_part = text.split(".")[-1].strip()

    # --------------------------------------------------
    # 2️⃣ Process ONLY the answer_part
    # --------------------------------------------------

    # 2.1 If there is '=', take number after last '='
    if "=" in answer_part:
        matches = re.findall(r"=\s*(-?\d+\.?\d*)", answer_part)
        if matches:
            return matches[-1]

    # 2.2 Look for <number> or <<number>>
    bracket_matches = re.findall(r"<+\s*(-?\d+\.?\d*)\s*>+", answer_part)
    if bracket_matches:
        return bracket_matches[-1]

    # 2.3 Fallback: last number in answer_part
    numbers = re.findall(r"-?\d+\.?\d*", answer_part)
    if numbers:
        return numbers[-1]

    return None


class data_building:
    def __init__(self, qa_path, llm_path, config):
        self.qa_data = pd.read_csv(qa_path)
        self.llm_description = loadjson(llm_path)
        self.llm_names = list(self.llm_description.keys())
        self.all_llm_description = [
            self.llm_description[name]['feature'] for name in self.llm_names
        ]
        self.MyLLMEngine = LLMEngine(
            llm_names=self.llm_names,
            llm_description=self.llm_description
        )
        self.config = config
        self.construct_data_with_LLM()

    def construct_data_with_LLM(self):

        df = pd.DataFrame(columns=[
            'task_id',
            'task_description',
            'task_description_embedding',
            'query',
            'query_embedding',
            'ground_truth_numeric',
            'ground_truth_reasoning',
            'model_response',
            'metric',
            'llm',
            'effect',
            'cost'
        ])

        for row in self.qa_data.itertuples():

            task_id_t = row.task_id
            query_t = row.query
            task_description = row.task_description
            metric_t = row.metric

            if task_id_t == "multi_news":
                tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
                tokens = tokenizer.tokenize(query_t)[:3000]
                query_t = tokenizer.convert_tokens_to_string(tokens)

            query_embedding = get_embedding([query_t])
            task_desc_embedding = get_embedding([task_description])

            # GSM8K-specific processing
            ground_truth_final = extract_gsm8k_final_answer(row.ground_truth)
            ground_truth_reasoning = extract_gsm8k_reasoning(row.ground_truth)

            for a_t, llm_name in enumerate(self.llm_names):

                response_t = self.MyLLMEngine.get_llm_response(
                    query=query_t,
                    llm_idx=a_t
                )

                if task_id_t == "GSM8K":
                    pred_answer = extract_final_answer_from_response(response_t)
                    reward_t = 1 if (
                        pred_answer is not None and
                        ground_truth_final is not None and
                        str(pred_answer).strip() == str(ground_truth_final).strip()
                    ) else 0
                else:
                    reward_t = self.MyLLMEngine.eval(
                        prediction=response_t,
                        ground_truth=row.ground_truth,
                        metric=metric_t
                    )

                cost_t = self.MyLLMEngine.compute_cost(
                    llm_idx=a_t,
                    input_text=query_t,
                    output_size=self.config['query_response_length']
                )

                df = df._append({
                    'task_id': task_id_t,
                    'task_description': task_description,
                    'task_description_embedding': task_desc_embedding,
                    'query': query_t,
                    'query_embedding': query_embedding,
                    'ground_truth_numeric': ground_truth_final,
                    'ground_truth_reasoning': ground_truth_reasoning,
                    'model_response': response_t,
                    'metric': metric_t,
                    'llm': llm_name,
                    'effect': reward_t,
                    'cost': cost_t
                }, ignore_index=True)

            break

        df['cost'] = df.groupby('task_id')['cost'].transform(
            lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8)
        )

        df.to_csv(os.path.join(self.config['data_dir'], "router_data.csv"), index=False)

        llm_description_embedding = get_embedding(self.all_llm_description)
        savepkl(
            llm_description_embedding,
            os.path.join(self.config['data_dir'], "llm_description_embedding.pkl")
        )

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    os.environ["KMP_DUPLICATE_LIB_OK"] = 'True'
    with open("configs/config.yaml", 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)

    load_dotenv()  # فایل .env را می‌خواند

    os.environ["TOGETHERAI_API_KEY"] = os.getenv("api_key")
    os.environ["OPENROUTER_API_KEY"] = config["OpenRouter_api_key"]


    data_dir = config['data_dir']
    data_building(
                qa_path=os.path.join(data_dir, 'unified_data.csv'),
                llm_path=os.path.join(data_dir, 'LLM_Descriptions.json'),
                config=config
                )