import numpy as np
import csv
import torch
import os
import time
import copy
import json
import torch.nn as nn
import torch.nn.functional as F
import torch
from sklearn.metrics import roc_auc_score


def ent(logits,dim):
    b = F.softmax(logits, dim) * F.log_softmax(logits, dim)
    return -1.0 * b.sum(dim,keepdim=True)

def compute_metrics(gt,scores):
    """
    Compute metrics based on ground truth values and scores.

    Args:
        gt (array-like): Ground truth values.
        scores (array-like): Scores.

    Returns:
        tuple: A tuple containing the computed metrics.
            - auc (float): The area under the ROC curve.
            - ce (float): The cross-entropy.
            - cestd (float): The cross-entropy standard deviation.

    """
    auc = roc_auc_score(gt, scores)
    # sgt = F.logsigmoid(scores * (gt * 2 - 1))
    # ce = -sgt.mean()
    sgt = F.binary_cross_entropy_with_logits(scores, gt.float(), reduction='none')
    ce = sgt.mean()
    cestd = sgt.std() / len(sgt) ** 0.5
    return auc, float(ce), float(cestd)

def prepare_inputs(inputs, device):
    for v in inputs:
        if isinstance(v, torch.Tensor):
            v = v.to(device)
    return inputs

## A simple MLP with n layers
## it takes input_dim, output_dim, hidden_dim, n_layers, activation function
## it returns a torch.nn.Sequential object
def make_mlp(input_dim, output_dim, hidden_dim, n_layers, activation, return_as_seq=True):
    layers = []
    layers.append(nn.Linear(input_dim, hidden_dim))
    layers.append(activation())
    for i in range(n_layers - 1):
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(activation())
    layers.append(nn.Linear(hidden_dim, output_dim))
    if return_as_seq:
        return nn.Sequential(*layers)
    return layers


## weight analysis
def analyze_tensor(w, nbins):
    """
    Calculate the quantiles of the input tensor w and its absolute values, and combines them into a single feature vector.
    The specific quantiles to compute are determined by the nbins variable.
    """

    if w is None:
        return torch.Tensor(nbins * 2).fill_(0)
    else:
        q = torch.arange(nbins).float() / (nbins - 1)
        hw = torch.quantile(w.view(-1).float(), q.to(w.device)).contiguous().cpu()
        hw_abs = torch.quantile(w.abs().view(-1).float(), q.to(w.device)).contiguous().cpu()
        fv = torch.cat((hw, hw_abs), dim=0)
        return fv


def extract_params_hist(model, nbins=100):
    with torch.no_grad():
        fvs = [analyze_tensor(p, nbins) for k, p in model.named_parameters()]
    return torch.stack(fvs, dim=0)


def extract_gradient_hist(model, example, target, loss_fn, nbins=100):
    model.eval()
    model.zero_grad()
    pred = model(example)
    loss = loss_fn(pred, target)
    loss.backward()
    gradients_hist = []
    for k, p in model.named_parameters():
        if p.grad:
            gradient = p.grad.data.clone()
            gradients_hist.append(analyze_tensor(gradient, nbins))
    return torch.stack(gradients_hist, dim=0)


## general metrics
def get_metrics(scores, gt):
    auc = roc_auc_score(gt, scores)
    sgt = F.logsigmoid(scores * (gt * 2 - 1))
    ce = -sgt.mean()
    cestd = sgt.std() / len(sgt) ** 0.5
    return auc, float(ce), float(cestd)


## show parameters by layers in the model
def show_params(model):
    total = 0
    for name, param in model.named_parameters():
        num = np.prod(param.shape)
        total += num
        shape = str(list(param.shape))
        print(f'{name:20s} {shape:20s} {num:10d}')
    print(f'Total number of parameters: {total}')


class NLP2023MetaNetwork(nn.Module):
    def __init__(self, raw_size=200, feat_size=60, hidden_size=22, nlayers_1=4, **kwargs):
        super().__init__()
        self.mlp_raw_to_feat = make_mlp(raw_size, feat_size, hidden_size, nlayers_1, nn.ReLU)
        self.mlp_feat_to_pred = make_mlp(2 * feat_size, 1, 30, 1, nn.ReLU)
        self.sigmoid = nn.Sigmoid()

    def intermediate(self, feat):
        h = torch.cat((torch.mean(feat, dim=0, keepdim=True), torch.std(feat, dim=0, keepdim=True)), dim=1)
        return h

    def forward(self, x):
        """
        :returns: a score for whether the network is a Trojan or not
        """
        out = [self.mlp_raw_to_feat(d['feats']) for d in x]
        out = [self.intermediate(h) for h in out]
        out = torch.cat(out)
        out = self.mlp_feat_to_pred(out)
        return self.sigmoid(out)

class NLP2023MetaNetwork_nointer(nn.Module):
    def __init__(self, raw_size=200, feat_size=60, hidden_size=22, nlayers_1=4, **kwargs):
        super().__init__()
        self.mlp_raw_to_feat = make_mlp(raw_size, feat_size, hidden_size, nlayers_1, nn.ReLU)
        self.mlp_feat_to_pred = make_mlp(feat_size, 1, 30, 1, nn.ReLU)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        :returns: a score for whether the network is a Trojan or not
        """
        out = [self.mlp_raw_to_feat(d['feats']) for d in x]
        out = [torch.mean(d,dim=0,keepdim=True) for d in out] #mean pool across episodes B, neps, 60 --> B, 1, 60
        out = torch.cat(out)
        out = self.mlp_feat_to_pred(out)
        return self.sigmoid(out)


class NLP2023MetaNetwork2(nn.Module):
    def __init__(self, raw_size=200, feat_size=60, hidden_size=22, nlayers_1=4, **kwargs):
        super().__init__()
        self.mlp_raw_to_feat = make_mlp(raw_size, feat_size, hidden_size, nlayers_1, nn.ReLU)
        # self.mlp_feat_to_pred = make_mlp(2 * feat_size + 1, 1, 30, 1, nn.ReLU) #adding reward at the end
        self.mlp_feat_to_pred = make_mlp(2 * feat_size, 1, 30, 1, nn.ReLU) #adding reward at the end
        self.sigmoid = nn.Sigmoid()

    def intermediate(self, feat):
        """
        Calculates the intermediate representation of the given feature tensor.

        Parameters:
            feat (torch.Tensor): The input feature tensor.

        Returns:
            torch.Tensor: The intermediate representation tensor.
        """
        h = torch.cat((torch.mean(feat, dim=0, keepdim=True), torch.std(feat, dim=0, keepdim=True)), dim=1)
        return h

    def forward(self, x):
        """
        :returns: a score for whether the network is a Trojan or not
        """
        out = [self.mlp_raw_to_feat(d['feats']) for d in x]
        out = [self.intermediate(h) for h in out]
        out = torch.cat(out)
        rewards = [ entr['reward'] for entr in x]
        rewards = torch.stack(rewards)
        # out = torch.cat((out, rewards), dim=1)
        out = self.mlp_feat_to_pred(out)
        return self.sigmoid(out)


class NLP2023_Meta_LogFeats(nn.Module):
    def __init__(self, raw_size=200, feat_size=60, hidden_size=22, nlayers_1=4, **kwargs):
        super().__init__()
        self.mlp_raw_to_feat = make_mlp(raw_size, feat_size, hidden_size, nlayers_1, nn.ReLU)
        self.mlp_feat_to_pred = make_mlp(2 * feat_size, 1, hidden_size, 1, nn.ReLU)
        self.sigmoid = nn.Sigmoid()

    def intermediate(self, feat):
        h = torch.cat((torch.mean(feat, dim=0, keepdim=True), torch.std(feat, dim=0, keepdim=True)), dim=1)
        return h

    def forward(self, x):
        """
        :returns: a score for whether the network is a Trojan or not
        """
        out = [self.mlp_raw_to_feat(d['feats']) for d in x]
        out = [self.intermediate(h) for h in out]
        out = torch.cat(out)
        out = self.mlp_feat_to_pred(out)
        return self.sigmoid(out)


class NLP2023MetaNetwork2b(nn.Module):
    def __init__(self, raw_size=200, feat_size=60, hidden_size=22, nlayers_1=4, **kwargs):
        super().__init__()
        self.mlp_raw_to_feat = make_mlp(raw_size, feat_size, hidden_size, nlayers_1, nn.ReLU)
        # self.mlp_feat_to_pred = make_mlp(2 * feat_size + 1, 1, 30, 1, nn.ReLU) #adding reward at the end
        self.mlp_feat_to_pred = make_mlp(feat_size, 1, 30, 1, nn.ReLU) #adding reward at the end
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        """
        :returns: a score for whether the network is a Trojan or not
        """
        # import pdb; pdb.set_trace()
        out = [self.mlp_raw_to_feat(d['feats']) for d in x] # MLP over all raw_features B, neps, 442 --> B, neps , 60
        out = [torch.mean(d,dim=0,keepdim=True) for d in out] #mean pool across episodes B, neps, 60 --> B, 1, 60
        out = torch.cat(out)
        rewards = [ entr['reward'] for entr in x]
        rewards = torch.stack(rewards)
        # out = torch.cat((out, rewards), dim=1)
        out = self.mlp_feat_to_pred(out)
        return self.sigmoid(out)
    

class NLP2023MetaNetwork2diff(nn.Module):
    def __init__(self, raw_size=200, feat_size=60, hidden_size=22, nlayers_1=4, **kwargs):
        super().__init__()
        self.raw_size = raw_size
        self.mlp_raw_to_feat = make_mlp(raw_size, feat_size, hidden_size, nlayers_1, nn.ReLU)
        # self.mlp_feat_to_pred = make_mlp(2 * feat_size + 1, 1, 30, 1, nn.ReLU) #adding reward at the end
        self.mlp_feat_to_pred = make_mlp(feat_size, 1, 30, 1, nn.ReLU) #adding reward at the end


    def forward(self, x):
        """
        :returns: a score for whether the network is a Trojan or not
        """
        # import pdb; pdb.set_trace()
        # out = [ d['feats'][:,:self.raw_size] - d['feats'][:,self.raw_size:] for d in x] # MLP over all raw_features B, neps, 442 --> B, neps , 60
        # out = [self.mlp_raw_to_feat(d) for d in out] # MLP over all raw_features B, neps, 442 --> B, neps , 60
        
        out = [self.mlp_raw_to_feat(d['feats']) for d in x] # MLP over all raw_features B, neps, 442 --> B, neps , 60
        out = [torch.mean(d,dim=0,keepdim=True) for d in out] #mean pool across episodes B, neps, 60 --> B, 1, 60
        out = torch.cat(out)
        rewards = [ entr['reward'] for entr in x]
        rewards = torch.stack(rewards)
        # out = torch.cat((out, rewards), dim=1)
        out = self.mlp_feat_to_pred(out)
        return out
    
def load_feature_dict(features, feature_name):
    episode = 0
    feats = features[episode]
    if feature_name == "both":
        fv1 = torch.stack(feats["value_grads"])
        fv2 = torch.stack(feats["action_grads"])
        fv = torch.cat((fv1.view(fv1.shape[0], -1), fv2.view(fv2.shape[0], -1)), dim=1)
    elif feature_name == "action_grads_value":
        fv1 = torch.FloatTensor(feats['values']).view(-1,1)
        fv2 = torch.stack(feats["action_grads"])
        fv = torch.cat((fv1, fv2.view(fv2.shape[0], -1)), dim=1)
    else:
        fv = torch.stack(feats[feature_name]) # 21, 7, 7 , 3

    feature_flat = fv.view(fv.shape[0], -1) # 21, 7*7*3
    final_reward = torch.FloatTensor([feats['final_reward']])
    return {"feats": torch.FloatTensor(feature_flat), "reward": final_reward}

def load_feature_dict_neps(features, feature_name):
    nepisodes = len(features)
    all_episode_features = []
    for episode in range(nepisodes):
        feats = features[episode]
        if feature_name == "value_grads_image":
            fv1 = torch.stack(feats["value_grads"])
            fv2 = feats["input_images"]
            fv = torch.cat((fv1.view(fv1.shape[0], -1), fv2.view(fv2.shape[0], -1)), dim=1)
        elif feature_name == "both":
            fv1 = torch.stack(feats["value_grads"])
            fv2 = torch.stack(feats["action_grads"])
            fv = torch.cat((fv1.view(fv1.shape[0], -1), fv2.view(fv2.shape[0], -1)), dim=1)
        elif feature_name == "both_diff":
            fv1 = torch.stack(feats["value_grads"])
            L  = len(fv1)
            fv2 = torch.stack(feats["action_grads"])
            fv3 = torch.stack(feats['logits']).view(L, -1)
            index = torch.argmax(fv3,1)
            fv2_chosen = []
            for i,ind in enumerate(index):
                fv2_chosen.append(fv2[i,ind,:,:,:])
            fv2_chosen = torch.stack(fv2_chosen)
            # diff = fv2_chosen - fv1
            # import pdb; pdb.set_trace()
            # print(fv1.shape, fv2.shape, fv3.shape)
            # fv = torch.cat((fv2_chosen.view(L, -1), fv1.view(L, -1)), dim=1)
            # fv = fv2_chosen.view(L, -1) -  fv1.view(L, -1)
            fv = fv2_chosen.view(L, -1) #only best action grads
            # fv = torch.mul(fv2_chosen, fv1).view(L, -1)

        elif feature_name == "action_grads_value":
            fv1 = torch.FloatTensor(feats['values']).view(-1,1)
            fv2 = torch.stack(feats["action_grads"])
            fv = torch.cat((fv1, fv2.view(fv2.shape[0], -1)), dim=1)
        elif feature_name == "action_grads_value_logits":
            fv1 = torch.FloatTensor(feats['values']).view(-1,1)
            L  = len(fv1)
            fv2 = torch.stack(feats["action_grads"])
            fv3 = torch.stack(feats['logits']).view(L, -1)
            # print(fv1.shape, fv2.shape, fv3.shape)
            fv = torch.cat((fv1, fv3, fv2.view(fv2.shape[0], -1)), dim=1)
        elif feature_name == "action_grads_value_entropy":
            fv1 = torch.FloatTensor(feats['values']).view(-1,1)
            fv2 = torch.stack(feats["action_grads"])
            L  = len(fv1)
            # import pdb; pdb.set_trace()
            fv3 = torch.stack(feats['logits']).view(L, -1)
            fv3_entropy = ent(fv3,dim=1)
            # print(fv1.shape, fv2.shape, fv3.shape)
            fv = torch.cat((fv1, fv3_entropy, fv2.view(fv2.shape[0], -1)), dim=1)
        elif feature_name == "action_grads_valuediff":
            fv1 = torch.stack(feats["value_grads"])
            L  = len(fv1)
            fv2 = torch.stack(feats["action_grads"])
            fv3_diff = fv2 - torch.unsqueeze(fv1,1)
            # print(fv1.shape, fv2.shape, fv3.shape)
            fv = fv3_diff.view(L, -1)
        else:
            fv = torch.stack(feats[feature_name]) # 21, 7, 7 , 3

        feature_flat = fv.view(fv.shape[0], -1) # 21, 7*7*3
        all_episode_features.append(feature_flat)
        final_reward = torch.FloatTensor([feats['final_reward']]) ##TODO: fix

    all_episode_features_flat = torch.cat(all_episode_features)
    return {"feats": all_episode_features_flat, "reward": final_reward}