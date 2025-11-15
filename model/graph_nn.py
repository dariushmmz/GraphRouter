import torch
import torch.nn.functional as F
from torch_geometric.nn import GeneralConv
from torch_geometric.data import Data
import torch.nn as nn
from torch.optim import AdamW
from sklearn.metrics import f1_score

class FeatureAlign(nn.Module):

    def __init__(self, query_feature_dim, llm_feature_dim, common_dim):
        super(FeatureAlign, self).__init__()
        self.query_transform = nn.Linear(query_feature_dim, common_dim)
        self.llm_transform = nn.Linear(llm_feature_dim, common_dim*2)
        self.task_transform = nn.Linear(llm_feature_dim, common_dim)

    def forward(self,task_id, query_features, llm_features):
        aligned_task_features = self.task_transform(task_id)
        aligned_query_features = self.query_transform(query_features)
        aligned_two_features=torch.cat([aligned_task_features,aligned_query_features], 1)
        aligned_llm_features = self.llm_transform(llm_features)
        aligned_features = torch.cat([aligned_two_features, aligned_llm_features], 0)
        return aligned_features


class EncoderDecoderNet(torch.nn.Module):

    def __init__(self, query_feature_dim, llm_feature_dim, hidden_features, in_edges):
        super(EncoderDecoderNet, self).__init__()
        self.in_edges = in_edges
        self.model_align = FeatureAlign(query_feature_dim, llm_feature_dim, hidden_features)
        self.encoder_conv_1 = GeneralConv(in_channels=hidden_features* 2, out_channels=hidden_features* 2, in_edge_channels=in_edges)
        self.encoder_conv_2 = GeneralConv(in_channels=hidden_features* 2, out_channels=hidden_features* 2, in_edge_channels=in_edges)
        self.edge_mlp = nn.Linear(in_edges, in_edges)
        self.bn1 = nn.BatchNorm1d(hidden_features * 2)
        self.bn2 = nn.BatchNorm1d(hidden_features * 2)

    def forward(self, task_id, query_features, llm_features, edge_index, edge_mask=None,
                edge_can_see=None, edge_weight=None):
        if edge_mask is not None:
            edge_index_mask = edge_index[:, edge_can_see]
            edge_index_predict = edge_index[:, edge_mask]
            if edge_weight is not None:
                edge_weight_mask = edge_weight[edge_can_see]
        edge_weight_mask=F.relu(self.edge_mlp(edge_weight_mask.reshape(-1, self.in_edges)))
        edge_weight_mask = edge_weight_mask.reshape(-1,self.in_edges)
        x_ini = (self.model_align(task_id, query_features, llm_features))
        x = F.relu(self.bn1(self.encoder_conv_1(x_ini, edge_index_mask, edge_attr=edge_weight_mask)))
        x = self.bn2(self.encoder_conv_2(x, edge_index_mask, edge_attr=edge_weight_mask))
        edge_predict = F.sigmoid(
            (x_ini[edge_index_predict[0]] * x[edge_index_predict[1]]).mean(dim=-1))
        return edge_predict


# جایگزین قسمت form_data.formulation در graph_nn.py
class form_data:

    def __init__(self,device):
        # توجه: device میتواند "cpu" یا "cuda"
        # پیشنهاد: برای ساخت تانسورها همیشه از cpu استفاده شود و سپس به GPU منتقل گردد.
        self.device = device

    def formulation(self,task_id,query_feature,llm_feature,org_node,des_node,edge_feature,label,edge_mask,combined_edge,train_mask,valide_mask,test_mask):
        """
        - task_id: np.array shape (num_queries, task_dim) or (1, task_dim)
        - query_feature: np.array shape (num_queries, query_dim) or (1, query_dim)
        - llm_feature: np.array shape (num_llms, llm_dim)
        - org_node: list of source node ids (length = num_edges)
        - des_node: list of dest node relative ids as used in repo (we keep original handling)
        - edge_feature: array shape (num_edges,) usually effect or similar
        - combined_edge: np.array shape (num_edges, k) where k is variable (cost,effect,feedback,...)
        - masks: arrays or lists of length num_edges
        """

        # --- STEP 1: create CPU numpy/torch arrays and basic checks ---
        import numpy as _np

        # Ensure numpy arrays
        query_features_np = _np.asarray(query_feature, dtype=_np.float32)
        llm_features_np = _np.asarray(llm_feature, dtype=_np.float32)
        task_id_np = _np.asarray(task_id, dtype=_np.float32)
        edge_feature_np = _np.asarray(edge_feature, dtype=_np.float32)
        combined_edge_np = _np.asarray(combined_edge, dtype=_np.float32)

        # Expand dims if necessary
        if query_features_np.ndim == 1:
            query_features_np = query_features_np.reshape(1, -1)
        if task_id_np.ndim == 1:
            task_id_np = task_id_np.reshape(1, -1)
        if llm_features_np.ndim == 1:
            llm_features_np = llm_features_np.reshape(1, -1)

        # number of queries and llms
        num_queries = query_features_np.shape[0]
        num_llms = llm_features_np.shape[0]

        # Compute expected number of edges if org_node/des_node provided as in repo
        # repo expects org_node like [0,0,0,...] for single query repeated num_llms times
        num_edges = len(org_node)
        if combined_edge_np.ndim == 1:
            combined_edge_np = combined_edge_np.reshape(-1, 1)

        # Basic sanity checks
        if combined_edge_np.shape[0] != num_edges:
            # try to broadcast or raise informative error
            if combined_edge_np.shape[0] == 1:
                combined_edge_np = _np.tile(combined_edge_np, (num_edges, 1))
            else:
                raise ValueError(f"combined_edge rows ({combined_edge_np.shape[0]}) != num_edges ({num_edges})")

        if edge_feature_np.shape[0] != num_edges:
            if edge_feature_np.shape[0] == 1:
                edge_feature_np = _np.tile(edge_feature_np, (num_edges,))
            else:
                raise ValueError(f"edge_feature length ({edge_feature_np.shape[0]}) != num_edges ({num_edges})")

        # Convert to torch tensors on CPU
        query_features = torch.tensor(query_features_np, dtype=torch.float)
        llm_features = torch.tensor(llm_features_np, dtype=torch.float)
        task_id = torch.tensor(task_id_np, dtype=torch.float)
        edge_weight = torch.tensor(edge_feature_np, dtype=torch.float).reshape(-1,1)
        combined_edge = torch.tensor(combined_edge_np, dtype=torch.float)  # shape: (num_edges, k)

        # create edge_index: note original code did des_node=[(i+1 + org_node[-1]) for i in des_node]
        # keep same semantics
        des_node_adj = [(i+1 + org_node[-1]) for i in des_node]
        edge_index = torch.tensor([org_node, des_node_adj], dtype=torch.long)

        # Ensure matching rows: if combined_edge rows > edge_weight rows, trim/raise
        if combined_edge.shape[0] != edge_weight.shape[0]:
            min_rows = min(combined_edge.shape[0], edge_weight.shape[0])
            combined_edge = combined_edge[:min_rows, :]
            edge_weight = edge_weight[:min_rows, :]

        # Concatenate leading edge_weight as first column (as repo used)
        combined_edge_final = torch.cat((edge_weight, combined_edge), dim=-1)  # shape (num_edges, 1+k)

        # convert masks to torch tensors (CPU) and ensure bool dtype
        edge_mask_t = torch.tensor(edge_mask, dtype=torch.bool) if edge_mask is not None else torch.ones(num_edges, dtype=torch.bool)
        train_mask_t = torch.tensor(train_mask, dtype=torch.bool) if train_mask is not None else torch.zeros(num_edges, dtype=torch.bool)
        valide_mask_t = torch.tensor(valide_mask, dtype=torch.bool) if valide_mask is not None else torch.zeros(num_edges, dtype=torch.bool)
        test_mask_t = torch.tensor(test_mask, dtype=torch.bool) if test_mask is not None else torch.zeros(num_edges, dtype=torch.bool)

        # Finally move to device (self.device) in one place to avoid async cuda asserts
        if str(self.device).startswith("cuda"):
            query_features = query_features.to(self.device)
            llm_features = llm_features.to(self.device)
            task_id = task_id.to(self.device)
            edge_index = edge_index.to(self.device)
            edge_weight = edge_weight.to(self.device)
            combined_edge_final = combined_edge_final.to(self.device)
            edge_mask_t = edge_mask_t.to(self.device)
            train_mask_t = train_mask_t.to(self.device)
            valide_mask_t = valide_mask_t.to(self.device)
            test_mask_t = test_mask_t.to(self.device)

        # prepare query_indices and llm_indices for possible downstream use
        query_indices = list(range(len(query_features)))
        llm_indices = [i + len(query_indices) for i in range(len(llm_features))]

        data = Data(
            task_id=task_id,
            query_features=query_features,
            llm_features=llm_features,
            edge_index=edge_index,
            edge_attr=edge_weight,  # keep original edge_attr as single-col effect (repo used)
            query_indices=query_indices,
            llm_indices=llm_indices,
            label=torch.tensor(label, dtype=torch.float).to(self.device) if label is not None else None,
            edge_mask=edge_mask_t,
            combined_edge=combined_edge_final,
            train_mask=train_mask_t,
            valide_mask=valide_mask_t,
            test_mask=test_mask_t
        )

        return data


class GNN_prediction:
    def __init__(self, query_feature_dim, llm_feature_dim,hidden_features_size,in_edges_size,wandb,config,device):

        self.model = EncoderDecoderNet(query_feature_dim=query_feature_dim, llm_feature_dim=llm_feature_dim,
                                        hidden_features=hidden_features_size,in_edges=in_edges_size).to(device)
        self.wandb = wandb
        self.config = config
        self.optimizer =AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        self.criterion = torch.nn.BCELoss()
        self.device = device

    def train_validate(self, data, data_validate, data_for_test):

        best_f1 = -1
        self.save_path = self.config['model_path']
        self.num_edges = len(data.edge_attr)

        self.train_mask = torch.tensor(data.train_mask, dtype=torch.bool)
        self.valide_mask = torch.tensor(data.valide_mask, dtype=torch.bool)
        self.test_mask = torch.tensor(data.test_mask, dtype=torch.bool)

        print("\n----- Training Started -----\n")

        for epoch in range(self.config['train_epoch']):
            self.model.train()
            loss_mean = 0
            mask_train = data.edge_mask

            # ------------------------
            #       TRAINING
            # ------------------------
            for inter in range(self.config['batch_size']):
                mask = mask_train.clone().bool().to(self.device)

                random_mask = (torch.rand(mask.size()) < self.config['train_mask_rate']).bool()
                random_mask = random_mask.to(self.device)
                mask = torch.where(mask & random_mask, torch.tensor(False), mask).bool()
                # mask = torch.where(mask & random_mask,
                                # torch.zeros(1, dtype=torch.bool, device=mask.device),
                                # mask).bool()

                edge_can_see = torch.logical_and(~mask, self.train_mask)

                self.optimizer.zero_grad()
                predicted_edges = self.model(
                    task_id=data.task_id,
                    query_features=data.query_features,
                    llm_features=data.llm_features,
                    edge_index=data.edge_index,
                    edge_mask=mask,
                    edge_can_see=edge_can_see,
                    edge_weight=data.combined_edge
                )

                loss = self.criterion(predicted_edges.reshape(-1), data.label[mask].reshape(-1))
                loss_mean += loss

            loss_mean = loss_mean / self.config['batch_size']
            loss_mean.backward()
            self.optimizer.step()

            # ------------------------
            #    VALIDATION
            # ------------------------
            self.model.eval()
            mask_validate = torch.tensor(data_validate.edge_mask, dtype=torch.bool)
            edge_can_see = self.train_mask

            with torch.no_grad():
                predicted_edges_validate = self.model(
                    task_id=data_validate.task_id,
                    query_features=data_validate.query_features,
                    llm_features=data_validate.llm_features,
                    edge_index=data_validate.edge_index,
                    edge_mask=mask_validate,
                    edge_can_see=edge_can_see,
                    edge_weight=data_validate.combined_edge
                )

                observe_edge = predicted_edges_validate.reshape(-1, self.config['llm_num'])
                observe_idx = torch.argmax(observe_edge, 1)

                value_validate = data_validate.edge_attr[mask_validate].reshape(-1, self.config['llm_num'])
                label_idx = torch.argmax(value_validate, 1)

                correct = (observe_idx == label_idx).sum().item()
                total = label_idx.size(0)
                validate_accuracy = correct / total

                f1 = f1_score(label_idx.cpu().numpy(), observe_idx.cpu().numpy(), average='macro')

                loss_validate = self.criterion(
                    predicted_edges_validate.reshape(-1),
                    data_validate.label[mask_validate].reshape(-1)
                )

            # ------------------------
            #   BEST MODEL SAVE
            # ------------------------
            if f1 > best_f1:
                best_f1 = f1
                torch.save(self.model.state_dict(), self.save_path)

            # ------------------------
            #        TEST
            # ------------------------
            test_result, test_loss = self.test(data_for_test, self.config['model_path'])

            # ------------------------
            #        PRINT METRICS
            # ------------------------
            print(f"\nEpoch {epoch+1}/{self.config['train_epoch']}")
            print(f"  Train Loss      : {loss_mean.item():.6f}")
            print(f"  Val Loss        : {loss_validate.item():.6f}")
            print(f"  Val Accuracy    : {validate_accuracy:.4f}")
            print(f"  Val Macro F1    : {f1:.4f}")
            print(f"  Test Result     : {test_result:.6f}")
            print(f"  Test Loss       : {test_loss.item():.6f}")
            print(f"  Best F1 so far  : {best_f1:.4f}")

            # ------------------------
            #     WANDB LOGGING
            # ------------------------
            self.wandb.log({
                "train_loss": loss_mean,
                "validate_loss": loss_validate,
                "validate_accuracy": validate_accuracy,
                "validate_f1": f1,
                "test_loss": test_loss,
                "test_result": test_result
            })

    def test(self,data,model_path):
        # self.model.load_state_dict(model_path)
        self.model.eval()
        mask = torch.tensor(data.edge_mask, dtype=torch.bool)
        edge_can_see = torch.logical_or(self.valide_mask, self.train_mask)
        with torch.no_grad():
            edge_predict = self.model(task_id=data.task_id,query_features=data.query_features, llm_features=data.llm_features, edge_index=data.edge_index,
                             edge_mask=mask,edge_can_see=edge_can_see,edge_weight=data.combined_edge)
        label = data.label[mask].reshape(-1)
        loss_test = self.criterion(edge_predict, label)
        edge_predict = edge_predict.reshape(-1, self.config['llm_num'])
        max_idx = torch.argmax(edge_predict, 1)
        value_test = data.edge_attr[mask].reshape(-1, self.config['llm_num'])
        label_idx = torch.argmax(value_test, 1)
        row_indices = torch.arange(len(value_test))
        result = value_test[row_indices, max_idx].mean()
        result_golden = value_test[row_indices, label_idx].mean()
        # print("result_predict:", result, "result_golden:",result_golden)

        return result,loss_test