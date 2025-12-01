import pandas as pd
from datasets import load_dataset
import yaml, os

task_descriptions = {
        "question-generation_PersianQA": """The question generation task assesses the ability of models to generate meaningful questions based on a given passage and answer. This tests the contextual and semantic understanding of the models in Persian.""",
        # "summarization_SamSUM-fa": """The summarization track tests how well models can generate concise and accurate summaries from Persian conversational or formal texts. Relevant columns include input text and human-written summaries.""",

    }

    
def generate_persian_qa_dataset(output_path='data/unified_qa_data.csv', sample_size=5000):
    """
    Convert Persian-NLG dataset into GraphRouter unified QA format.

    Dataset fields:
        - context
        - question
        - answer
    """

    # print("Loading Persian dataset ...")
    # ds = load_dataset("MCINext/persian-nlg", split="train")

    import json

    path = "data/MCINext_persian_nlg"
    rows = []
    tokenC = 0

    for task in task_descriptions.keys():
        taskName = task
        task = task+'.jsonl'
        file_path = f"{path}/{task}"
        print("=======task=======>", taskName)
        
        with open(file_path, "r", encoding="utf-8") as f:

            for i, item in enumerate(f):
                data = json.loads(item)

                context = data["context"]
                question = data["question"]
                answer = data["answer"]

                # Build query text
                query = context.strip() + "\n\n" + question.strip()

                tokenC += len(query.split(' '))

                rows.append({
                    "task_id": taskName,
                    "query": query,
                    "ground_truth": answer,
                    "metric": "f1_score",
                    "task_description": task_descriptions[taskName]
                })

    print(tokenC)
    # Convert to DataFrame
    df = pd.DataFrame(rows)

    # Save
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Saved unified QA dataset to {output_path} ({len(df)} rows)")



if __name__ == "__main__":
    with open("configs/config.yaml", 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)

    generate_persian_qa_dataset(
        output_path=config["unified_qa_data_path"],
        sample_size=config.get("persian_sample_size")
    )

