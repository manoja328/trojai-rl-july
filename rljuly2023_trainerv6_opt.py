import torch
import os
import pandas as pd
from dataclasses import dataclass
import copy
import torch.nn as nn
import numpy as np
from scipy import stats
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import wandb

from utils.trojai_utils import *
from rljuly2023_featuresv5 import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict(saved_config, raw_feat):
    scores = []
    with torch.no_grad():
        kfold_state_dicts = saved_config.pop("state_dicts")
        for fold_state in kfold_state_dicts:
            ## remaining is config params
            model = NLP2023MetaNetwork2b(**saved_config)
            model = model.load_state_dict(fold_state['state_dict'])
            raw_feat = extract_params_hist(model)
            score = model(raw_feat).squeeze().item()
            scores.append(score)
    # majority voting
    trojan_probability = stats.mode(scores).mode
    return float(trojan_probability)


def custom_collate(batch):
    return [x[0] for x in batch], torch.LongTensor([x[1] for x in batch])

class TrojFeatures(Dataset):
    def __init__(self, data_df, feature_name="value_grads"):
        self.data = data_df
        self.feature_name = feature_name

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        SAVE_PATH = os.path.join(FEATS_DIR, f'{row.model_name}.pt')
        all_features = torch.load(SAVE_PATH)
        # # ## feature is nxd where n is the number of layers and d is the feature dimension
        # # ## get the first 100 eigenvales after SVD
        # svd.fit(features)
        # features = svd.singular_values_[:80]
        # features = torch.FloatTensor(features).unsqueeze(0)
        # features = features / features.norm(2)
        fvs = load_feature_dict_neps(all_features, self.feature_name)
        # feature = torch.cat((feature, final_reward.unsqueeze(1)), dim=1)
        label = int(row.ground_truth == "triggered")
        return fvs, label

    def __len__(self):
        return len(self.data)

def forward_step(model, batch):
    inputs, targets = batch
    inputs_to_device = prepare_inputs(inputs, device)
    outputs = model(inputs_to_device)
    outputs = outputs.squeeze(1)
    return outputs


def evaluate(model, dl):
    model.eval()
    pred = []
    gt = []
    current_loss = 0
    with torch.no_grad():
        for i, data in enumerate(dl):
            inputs, targets = data
            outputs = forward_step(model, data)
            loss = F.binary_cross_entropy_with_logits(outputs, targets.float().to(device))
            current_loss += loss.item()
            gt.extend(targets.long().tolist())
            pred.extend(outputs.tolist())

    test_loss = current_loss / len(dl)
    return test_loss, torch.Tensor(pred), torch.Tensor(gt).long()

global_best_auc = 0
def train(config=None):
    global global_best_auc
    print("Training...")
    with wandb.init(config=config):
        config = wandb.config
        print(config)
        meta_df = get_metadata()
        # meta_df.iloc[0].poisoned
        ## k fold CV
        skf = StratifiedKFold(n_splits=4)
        kfolds = skf.split(meta_df, meta_df.ground_truth)
        state_dicts = []
        splits_auc = []
        splits_ce = []
        for split_id, (train_index, test_index) in enumerate(kfolds):
            print(f"Fold {split_id}:")
            print(f"  Train: index={train_index}")
            print(f"  Test:  index={test_index}")

            X_train = meta_df.iloc[train_index]
            X_test = meta_df.iloc[test_index]
            # copy the test set to val set
            X_val = X_test

            model = NLP2023MetaNetwork2diff(raw_size= config.raw_size, feat_size=config.feat_size,
                                         hidden_size=config.hidden_size, nlayers_1=config.nlayers_1)
                        
            model = model.to(device)
            if split_id == 0:
                print(model)

            optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

            train_ds = TrojFeatures(X_train,config.feature_name)
            # also from train
            val_ds = TrojFeatures(X_val,config.feature_name)
            ## real test
            test_ds = TrojFeatures(X_test,config.feature_name)

            print("train ", X_train.ground_truth.value_counts())
            print("val ", X_val.ground_truth.value_counts())
            print("test ", X_test.ground_truth.value_counts())

            trainloader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, collate_fn=custom_collate, num_workers=4)
            valloader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=custom_collate)
            testloader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=custom_collate)

            best_auc = 0
            for epoch in range(config.epochs):
                current_loss = 0.0
                model.train()
                for i, data in enumerate(trainloader):
                    inputs, targets = data
                    optimizer.zero_grad()
                    outputs = forward_step(model, data)
                    loss = F.binary_cross_entropy_with_logits(outputs, targets.float())

                    ## L2 regularization
                    l2 = torch.stack([(p ** 2).sum() for p in model.parameters()], dim=0).sum()
                    loss = loss + config.decay * l2

                    ## L1 regularization
                    # l1 = torch.stack([p.abs().sum() for p in model.parameters()],dim=0).sum()
                    # loss = loss + config.decay*l1

                    loss.backward()
                    optimizer.step()
                    current_loss += loss.item()
                    if i + epoch == 0:
                        print(f"first loss {current_loss:.4f}")

                avgval_loss, pred, gt = evaluate(model, valloader)
                val_auc = roc_auc_score(gt.numpy(), torch.sigmoid(pred).cpu().numpy())
                if val_auc > best_auc:
                    # save the state dict
                    best_auc = val_auc
                    save_dict = {"state_dict": copy.deepcopy(model.state_dict()),
                                 "auc": val_auc,
                                }

                avgtrain_loss = current_loss / len(trainloader)
                print(f'Epoch {epoch:3d} train_loss {avgtrain_loss:.4f} val_loss {avgval_loss:.4f} AUC: {val_auc:.4f}')

            #Temperature-scaling calibration
            model.load_state_dict(save_dict['state_dict'])
            model.eval()
            
            avgtest_loss, uncalibrated_scores, gt = evaluate(model, testloader)

            T = torch.Tensor(1).fill_(0)
            T.requires_grad_()
            temp_optimizer = torch.optim.Adamax([T],lr=3e-2)
            #TODO: do a early stopping or plteau here
            for iter_idx in range(500):
                temp_optimizer.zero_grad()
                loss=F.binary_cross_entropy_with_logits(uncalibrated_scores*torch.exp(-T),gt.float())
                loss.backward()
                temp_optimizer.step()
                if iter_idx % 100 == 0:
                    print(f"iter {iter_idx} loss {loss:.4f} T {T.item()}")

            # import pdb; pdb.set_trace()
            calibrated_scores = uncalibrated_scores * torch.exp(-T).detach()

            auc_0, ce, cestd = compute_metrics(gt, uncalibrated_scores)
            print(f"Un-Calibrated Split {split_id}, auc: {auc_0:.4f}, ce: {ce:.4f}, cestd: {cestd:.4f}")
            auc_1, ce, cestd = compute_metrics(gt, calibrated_scores)
            print(f"   Calibrated Split {split_id}, auc: {auc_1:.4f}, ce: {ce:.4f}, cestd: {cestd:.4f}")

            # save to the best ensemble
            save_dict["T"] = T.item()
            state_dicts.append(save_dict)

            splits_auc.append(auc_1)
            splits_ce.append(ce)


        ## show all best acc of all folds
        best_val_aucs = [x['auc'] for x in state_dicts]
        print("config.feature_name: ", config.feature_name)
        print(f"kfold best val aucs: {best_val_aucs} and mean: {np.mean(best_val_aucs):.4f}")

        log_dict = {"val_auc": np.mean(splits_auc), "val_ce": np.mean(splits_ce)}
        wandb.log(log_dict)
        final_save_dict = {"state_dicts": state_dicts}
        cf = config._as_dict()
        cf.pop("_wandb")
        final_save_dict.update(cf)
        torch.save(final_save_dict, f"{config.feature_name}_v6.pth")
        if np.mean(splits_auc) > global_best_auc:
            global_best_auc = np.mean(splits_auc)
            torch.save(final_save_dict, f"global_best_{config.feature_name}_v6.pth")


# sweep_configuration = {
#     'method': 'grid', #grid, random
#     'metric': {
#         'goal': 'maximize',
#         'name': 'val_auc'
#         },
#     'parameters': {
#         'nlayers_1': {'values': [1,2,3]},
#         'epochs': {'values': [10,20,30,40]},
#         # 'decay': { 'distribution':'log_uniform_values', 'min': 1e-8, 'max':1e-2},
#         'decay':  {'value': 0.0001},
#         'lr': {'values': [0.01,0.001,0.0001]},
#      }
# }

# https://docs.wandb.ai/guides/sweeps/define-sweep-configuration
sweep_configuration = {
    'method': 'bayes',
    'metric': {
        'goal': 'maximize',
        'name': 'val_auc'
    },
    'parameters': {
        'nlayers_1': {'values': [1,2,3]},
        'feature_name': {'value': "action_grads_value_entropy"},
        'raw_size': {'value': 443},
        'batch_size': {'values': [4,8,16,32]},
        'feat_size': {'distribution':'q_log_uniform_values', 'min': 16, 'max':128},
        'hidden_size': {'distribution':'q_log_uniform_values', 'min': 16, 'max':128},
        'epochs': {'distribution':'q_log_uniform_values', 'min': 10, 'max':60},
        'decay': { 'distribution':'log_uniform_values', 'min': 1e-8, 'max':1e-2},
        'lr': { 'distribution':'log_uniform_values', 'min': 1e-4, 'max':1e-2},
    }
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project='rl_backdoor')
max_runs = 50
wandb.agent(sweep_id, function=train, count=max_runs)