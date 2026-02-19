import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import random
import numpy as np
import torch
from graph_nn import  form_data, EncoderDecoderNet
from data_processing.utils import ask_and_save_feedback, ensure_2d, parse_embedding_field
import pandas as pd
import json
import pickle
import re
import yaml
device = "cuda" if torch.cuda.is_available() else "cpu"
print("---------------> ALL IMPORTED")


 
def make_plot(scores, output_dir, scenario, query_id, task_id):
    results_list = []
    for llm, score in scores.items():
        results_list.append({
            "LLM": llm,
            "Score": score,
            "Scenario": scenario
        })

    df_scores = pd.DataFrame(results_list)
    import plotly.io as pio
    import plotly.express as px
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
        title=f"LLM Scores for {scenario} Scenario for Query {query_id} | Task {task_id}"
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
    try:
        fig.write_html(output_file)
        print(f"Graph saved as: {output_file}")
        pio.renderers.default = "browser"

    except:
        pio.renderers.default = "colab"

    fig.show()
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


def savejson(data: dict, filename: str) -> None:
    """
    Save data to a JSON file.

    Args:
        data: Dictionary to save
        filename: Path where the JSON file will be saved
    """
    with open(filename, 'w') as json_file:
        json.dump(data, json_file, indent=4)


def loadpkl(filename: str) -> any:
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


def savepkl(data: any, filename: str) -> None:
    """
    Save data to a pickle file.

    Args:
        data: Object to save
        filename: Path where the pickle file will be saved
    """
    with open(filename, 'wb') as pkl_file:
        pickle.dump(data, pkl_file)

class graph_router_prediction:
    def __init__(self, router_data_path,llm_path,llm_embedding_path,config, wandb, query_id=0,task_id=None,adaptive=False, inference=False):
        self.config = config
        self.wandb = wandb
        self.data_df = pd.read_csv(router_data_path)
        if task_id:
            self.data_df[self.data_df['task_id'] == task_id]
        self.llm_description = loadjson(llm_path)
        self.llm_names = list(self.llm_description.keys())
        self.num_llms=len(self.llm_names)
        self.num_query=int(len(self.data_df)/self.num_llms)
        self.num_task=config['num_task']
        self.set_seed(self.config['seed'])
        self.llm_description_embedding=loadpkl(llm_embedding_path)
        self.llm_dim = self.llm_description_embedding.shape[1]


        if inference:
            self.num_llms

            start = int(query_id) * self.num_llms
            rows = self.data_df.iloc[start:start + self.num_llms]
            print(f"\n================Task: {rows.iloc[0]['task_id']}================")


            self.effect_list = np.array(rows['effect'].tolist())
            self.cost_list = np.array(rows['cost'].tolist())

            query_embedding = rows.iloc[0]['query_embedding']
            task_embedding = rows.iloc[0]['task_description_embedding']

            query_embedding = ensure_2d(parse_embedding_field(query_embedding))
            task_embedding = ensure_2d(parse_embedding_field(task_embedding))
            

            
            query_embedding = np.array(query_embedding)
            task_embedding = np.array(task_embedding)

            query_dim = query_embedding.shape[1]

            self.model = EncoderDecoderNet(query_feature_dim=query_dim, llm_feature_dim=self.llm_dim,
                                                hidden_features=self.config['embedding_dim'],in_edges=self.config['edge_dim']).to(device)
            self.form_data = form_data(device)
            results = self.infer_single_query(query_embedding, task_embedding)
            
            outpath = f"data/results/{rows.iloc[0]['task_id']}"
            make_plot(results['scores'], output_dir=outpath, 
                scenario=self.config['scenario'],
                query_id=query_id, task_id=rows.iloc[0]['task_id'])



            if adaptive:
                feedback_path = f"{self.config['data_dir']}/feedback.json"
                feedback_score = ask_and_save_feedback(query_id, results['best_llm'], feedback_path)

                from data_processing.delayed_reward import AdaptiveBatchUpdater

                adaptive_updater = AdaptiveBatchUpdater(
                router_csv_path=router_data_path,
                llm_description_path=llm_path,
                batch_size=10,
                eta=0.05,
                use_feedback=True,
                beta=0.3
                )

                adaptive_updater.compute(rows.iloc[0]['query'],
                    feedback_score, 
                    results['best_llm'], 
                    query_id,
                    self.num_llms)

            
                adaptive_updater.flush()




        else:
            from graph_nn import GNN_prediction
            self.prepare_data_for_GNN()
            self.split_data()
            self.form_data = form_data(device)
            self.query_dim = self.query_embedding_list.shape[1]
            self.GNN_predict = GNN_prediction(query_feature_dim=self.query_dim, llm_feature_dim=self.llm_dim,
                                        hidden_features_size=self.config['embedding_dim'], in_edges_size=self.config['edge_dim'],wandb=self.wandb,config=self.config,device=device)
            print("GNN training successfully initialized.")
            self.train_GNN()


    def set_seed(self,seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def split_data(self):
        self.query_per_task=int(self.num_query/self.num_task)
        split_ratio = self.config['split_ratio']

        # Calculate the size of each set for each task
        train_size = int(self.query_per_task * split_ratio[0])
        val_size = int(self.query_per_task * split_ratio[1])
        test_size = int(self.query_per_task * split_ratio[2])

        # Generate indices
        train_idx = []
        validate_idx = []
        test_idx = []

        for task_id in range(self.num_task):
            # Starting index for each task
            start_idx = task_id * self.query_per_task * self.num_llms

            # Add training set indices
            train_idx.extend(range(start_idx, start_idx + train_size* self.num_llms))

            # Add validation set indices
            validate_idx.extend(range(start_idx + train_size* self.num_llms,
                                      start_idx + train_size* self.num_llms + val_size* self.num_llms))

            # Add test set indices
            test_idx.extend(range(start_idx + train_size* self.num_llms + val_size* self.num_llms,
                                  start_idx + train_size* self.num_llms + val_size* self.num_llms + test_size* self.num_llms))


        self.effect_list = (self.effect_list -self.effect_list.min()) / (self.effect_list.max() - self.effect_list.min() + 1e-12)
        self.cost_list = (self.cost_list - self.cost_list.min()) / (self.cost_list.max() - self.cost_list.min() + 1e-12)

        self.combined_edge=np.concatenate((self.cost_list.reshape(-1,1),self.effect_list.reshape(-1,1)),axis=1)
        self.scenario=self.config['scenario']
        if self.scenario== "Performance First":
            self.effect_list = 1.0 * self.effect_list - 0.0 * self.cost_list
        elif self.scenario== "Balance":
            self.effect_list = 0.5 * self.effect_list - 0.5 * self.cost_list
        else:
            self.effect_list = 0.2 * self.effect_list - 0.8 * self.cost_list

        # reshape to (num_queries, num_llms)
        weighted_score_reshaped = self.effect_list.reshape(-1, self.num_llms)
        # temperature scaling
        SOFTMAX_TEMPERATURE = 0.15
        weighted_score_reshaped = weighted_score_reshaped / SOFTMAX_TEMPERATURE
        # numerically stable softmax
        exp_scores = np.exp(
            weighted_score_reshaped - np.max(weighted_score_reshaped, axis=1, keepdims=True)
        )
        softmax_per_query = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        # flatten back
        self.effect_list = softmax_per_query.reshape(-1)


        effect_re=self.effect_list.reshape(-1,self.num_llms)
        self.label=np.eye(self.num_llms)[np.argmax(effect_re, axis=1)].reshape(-1,1)
        self.edge_org_id=[num for num in range(self.num_query) for _ in range(self.num_llms)]
        self.edge_des_id=list(range(self.edge_org_id[0],self.edge_org_id[0]+self.num_llms))*self.num_query

        self.mask_train =torch.zeros(len(self.edge_org_id))
        self.mask_train[train_idx]=1

        self.mask_validate = torch.zeros(len(self.edge_org_id))
        self.mask_validate[validate_idx] = 1

        self.mask_test = torch.zeros(len(self.edge_org_id))
        self.mask_test[test_idx] = 1


    def prepare_data_for_GNN(self):
        unique_index_list=list(range(0, len(self.data_df), self.num_llms))
        query_embedding_list_raw=self.data_df['query_embedding'].tolist()
        task_embedding_list_raw = self.data_df['task_description_embedding'].tolist()

        self.query_embedding_list= []
        self.task_embedding_list= []
        for inter in query_embedding_list_raw:
            q_emb = parse_embedding_field(inter)
            self.query_embedding_list.append(q_emb)

        for inter in task_embedding_list_raw:
            t_emb = parse_embedding_field(inter)
            self.task_embedding_list.append(t_emb)

            
        self.query_embedding_list=np.array(self.query_embedding_list)[unique_index_list]
        self.task_embedding_list = np.array(self.task_embedding_list)[unique_index_list]
        self.effect_list=np.array(self.data_df['effect'].tolist())
        self.cost_list=np.array(self.data_df['cost'].tolist())



    def train_GNN(self):

        self.data_for_GNN_train = self.form_data.formulation(task_id=self.task_embedding_list,
                                                             query_feature=self.query_embedding_list,
                                                             llm_feature=self.llm_description_embedding,
                                                             org_node=self.edge_org_id,
                                                             des_node=self.edge_des_id,
                                                             edge_feature=self.effect_list, edge_mask=self.mask_train,
                                                             label=self.label, combined_edge=self.combined_edge,
                                                             train_mask=self.mask_train, valide_mask=self.mask_validate,
                                                             test_mask=self.mask_test)
        self.data_for_GNN_validate = self.form_data.formulation(task_id=self.task_embedding_list,
                                                                query_feature=self.query_embedding_list,
                                                                llm_feature=self.llm_description_embedding,
                                                                org_node=self.edge_org_id,
                                                                des_node=self.edge_des_id,
                                                                edge_feature=self.effect_list,
                                                                edge_mask=self.mask_validate, label=self.label,
                                                                combined_edge=self.combined_edge,
                                                                train_mask=self.mask_train,
                                                                valide_mask=self.mask_validate,
                                                                test_mask=self.mask_test)

        self.data_for_test = self.form_data.formulation(task_id=self.task_embedding_list,
                                                        query_feature=self.query_embedding_list,
                                                        llm_feature=self.llm_description_embedding,
                                                        org_node=self.edge_org_id,
                                                        des_node=self.edge_des_id,
                                                        edge_feature=self.effect_list, edge_mask=self.mask_test,
                                                        label=self.label, combined_edge=self.combined_edge,
                                                        train_mask=self.mask_train, valide_mask=self.mask_validate,
                                                        test_mask=self.mask_test)
        self.GNN_predict.train_validate(data=self.data_for_GNN_train, data_validate=self.data_for_GNN_validate,data_for_test=self.data_for_test)

    def test_GNN(self):
        predicted_result = self.GNN_predict.test(data=self.data_for_test,model_path=self.config['model_path'])


    def infer_single_query(self, query_embedding, task_embedding=None):
        """
        Perform inference for ONE query.
        
        Args:
            query_embedding: numpy array (query_dim,)
            task_embedding: numpy array (task_dim,) (if None, use first task)
            
        Returns:
            dict with scores and best llm
        """

        # Load best model
        self.model.load_state_dict(
            torch.load(self.config['model_path'], map_location=device)
        )
        self.model.eval()

        # If only one task in your setup
        if task_embedding is None:
            task_embedding = self.task_embedding_list[0]

        query_embedding = np.array(query_embedding).reshape(1, -1)
        task_embedding = np.array(task_embedding).reshape(1, -1)

        num_llms = self.num_llms

        # ---- Construct Graph ----

        # Query → LLM edges
        org_node = [0] * num_llms
        des_node = list(range(num_llms))  # will be shifted inside formulation

        # Dummy edge feature (effect) — must match shape
        # edge_feature = np.zeros(num_llms)

        # combined_edge must have shape (num_edges, 2)
        # combined_edge = [effect, cost]
        # combined_edge = np.zeros((num_llms, 2))
        self.effect_list = (self.effect_list -self.effect_list.min()) / (self.effect_list.max() - self.effect_list.min() + 1e-12)
        self.cost_list = (self.cost_list - self.cost_list.min()) / (self.cost_list.max() - self.cost_list.min() + 1e-12)

        combined_edge=np.concatenate((self.cost_list.reshape(-1,1),self.effect_list.reshape(-1,1)),axis=1)
        
        self.scenario=self.config['scenario']
        if self.scenario== "Performance First":
            self.effect_list = 1.0 * self.effect_list - 0.0 * self.cost_list
        elif self.scenario== "Balance":
            self.effect_list = 0.5 * self.effect_list - 0.5 * self.cost_list
        else:
            self.effect_list = 0.2 * self.effect_list - 0.8 * self.cost_list


        # reshape to (num_queries, num_llms)
        weighted_score_reshaped = self.effect_list.reshape(-1, self.num_llms)
        # temperature scaling
        SOFTMAX_TEMPERATURE = 0.15
        weighted_score_reshaped = weighted_score_reshaped / SOFTMAX_TEMPERATURE
        # numerically stable softmax
        exp_scores = np.exp(
            weighted_score_reshaped - np.max(weighted_score_reshaped, axis=1, keepdims=True)
        )
        softmax_per_query = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        # flatten back
        self.effect_list = softmax_per_query.reshape(-1)


        # Masks (we predict ALL edges)
        edge_mask = torch.ones(num_llms).to(device)
        train_mask = torch.ones(num_llms).to(device)
        validate_mask = torch.zeros(num_llms).to(device)
        test_mask = torch.zeros(num_llms).to(device)

        # Dummy label (not used)
        dummy_label = np.zeros((num_llms, 1))

        # ---- Build Data ----
        data = self.form_data.formulation(
            task_id=task_embedding,
            query_feature=query_embedding,
            llm_feature=self.llm_description_embedding,
            org_node=org_node,
            des_node=des_node,
            edge_feature=self.effect_list,
            label=dummy_label,
            edge_mask=edge_mask,
            combined_edge=combined_edge,
            train_mask=train_mask,
            valide_mask=validate_mask,
            test_mask=test_mask
        )

        # During inference, allow full visibility
        edge_can_see = torch.ones(num_llms).bool().to(device)

        # ---- Forward ----
        with torch.no_grad():
            edge_scores = self.model(
                task_id=data.task_id,
                query_features=data.query_features,
                llm_features=data.llm_features,
                edge_index=data.edge_index,
                edge_mask=edge_mask.bool(),
                edge_can_see=edge_can_see,
                edge_weight=data.combined_edge
            )
        pred = edge_scores.reshape(-1, self.num_llms)
        T = .001
        pred = torch.softmax(pred / T, dim=1)

        # edge_scores = edge_scores.cpu().numpy().reshape(-1)
        best_idx = torch.argmax(pred, 1)

        gt_scores = data.edge_attr.reshape(-1, self.num_llms)
        gt_idx = torch.argmax(gt_scores, 1)



        # Top-3 predicted indices
        edge_scores = pred[0].cpu().numpy()
        top3_pred_idx = np.argsort(edge_scores)[::-1][:3]
        top3_pred_scores = edge_scores[top3_pred_idx]


        # If GT is one-hot per query:
        gt_scores = gt_scores[0].cpu().numpy()
        top3_gt_idx = np.argsort(gt_scores)[::-1][:3]
        top3_gt_scores = gt_scores[top3_gt_idx]

        print("Top-3 Predicted LLMs:")
        for rank, (idx, score) in enumerate(zip(top3_pred_idx, top3_pred_scores), 1):
            print(f"{rank}. LLM {self.llm_names[idx]:<3} | Score: {score:.4f}")

        print("\nTop-3 Ground Truth LLMs:")
        for rank, (idx, score) in enumerate(zip(top3_gt_idx, top3_gt_scores), 1):
            print(f"{rank}. LLM {self.llm_names[idx]:<3} | Score: {score:.4f}")

        print("\n--------------------------------------------")

        gt_scores = [round(float(s), 3) for s in gt_scores]


        scores = {
            self.llm_names[i]: float(pred[0, i].cpu().item())
            for i in range(len(self.llm_names))
        }
        
        return {
            "best_llm": self.llm_names[best_idx],
            "scores": scores,
            "edge_scores": edge_scores,
            'ground_truth': self.llm_names[gt_idx],
            'gt_scores': gt_scores

        }




if __name__ == "__main__":
    import wandb
    with open("configs/config.yaml", 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    wandb_key = config['wandb_key']
    wandb.login(key=wandb_key)
    wandb.init(project="graph_router")

    
    data_dir = config['data_dir']
    router_data_path = os.path.join(data_dir, 'router_data.csv')
    if config['feedback']:
        router_data_path = os.path.join(data_dir, 'feedback/router_data.csv')
        if not os.path.exists(router_data_path):
            print("[INFO] No feedback found")
            router_data_path = os.path.join(data_dir, 'router_data.csv')

    graph_router_prediction(
        
        router_data_path=router_data_path,
        llm_path=os.path.join(data_dir, 'LLM_Descriptions.json'),
        llm_embedding_path=os.path.join(data_dir, "llm_description_embedding.pkl"),
        config=config,
        wandb=wandb
    )