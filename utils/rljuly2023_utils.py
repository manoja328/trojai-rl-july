import numpy as np
import csv
import os
import copy
import json
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.autograd.functional as AF
import torch_ac
import gym
from gym_minigrid.wrappers import ImgObsWrapper
preprocess = torch_ac.format.default_preprocess_obss

class RL2023july:
    def __init__(self, model_filepath, device="cpu"):
        self.root = model_filepath.rstrip("/model.pt")
        self.device = device
        self.model = torch.load(model_filepath).to(device)
        self.model.eval()

    def istrojaned(self):
        with open(os.path.join(self.root, 'ground_truth.csv'), 'r') as f:
            for line in f:
                label = int(line)
                break
        return label

def root():
    return '/workspace/manoj/trojai-datasets/rl-lavaworld-jul2023'


def load_engine(MODEL_ID, device="cpu"):
    model_filepath = os.path.join(root(), 'models', 'id-%08d' % MODEL_ID, 'model.pt')
    if os.path.exists(model_filepath):
        return RL2023july(model_filepath, device)
    else:
        raise FileNotFoundError(f"folder {model_filepath} not found")

def get_metadata():
    import pandas as pd
    path = os.path.join(root(), 'METADATA.csv')
    return pd.read_csv(path)

def get_jacobian(model, obs):
    with torch.no_grad():
        dist, value = model(obs)
    def func_act(x):
        return model(x)[0].logits.squeeze(dim=0)
    def func_val(x):
        return model(x)[1].squeeze(dim=0)
    
    # import pdb; pdb.set_trace()
    with torch.set_grad_enabled(True):
        action_grad = AF.jacobian(func_act,obs.float())
        value_grad = AF.jacobian(func_val,obs.float())

    return dist, value, action_grad, value_grad

def get_jacobian_features(env_string, model, device="cpu"):
    # Run episodes through an environment to collect what may be relevant information to trojan detection
    # Construct environment and put it inside a observation wrapper
    env = ImgObsWrapper(gym.make(env_string))

    episodes = 10
    all_features = []
    final_rewards = []
    max_episode_length = 100
    # Episode loop
    for _ in range(episodes):
        # Reset environment after episode and get initial observation
        obs = env.reset()
        done = False
        # Per episode loop
        value_grads = []
        action_grads = []
        logits = []
        values = []
        frame_number = 0
        while not done and frame_number < max_episode_length:
            # Preprocessing function to prepare observation from env to be given to the model
            obs = preprocess([obs], device=device)
            # Use env observation to get action distribution
            dist, value, action_grad, value_grad = get_jacobian(model, obs)
            
            # (Pdb) action_grad.shape
            # torch.Size([3, 1, 7, 7, 3])
            # (Pdb) value_grad.shape
            # torch.Size([1, 7, 7, 3])
            # import pdb; pdb.set_trace()
            # (Pdb) dist.logits
            # tensor([[-1.9621, -0.1904, -3.4161]])
            # (Pdb) value
            # tensor([0.7015])

            # Sample from distribution to determine which action to take
            action = dist.sample()
            action = action.cpu().detach().numpy()
            values.append(value.item())
            # Use action to step environment and get new observation
            obs, reward, done, info = env.step(action)
            action_grads.append(action_grad.squeeze().cpu().detach())
            value_grads.append(value_grad.squeeze().cpu().detach())
            logits.append(dist.logits)
            frame_number += 1
        
        if reward == 0:
            print(f"dead in {frame_number} steps")
        else:
            print(f"goal acheived in {frame_number} steps")

        features = {"value_grads": value_grads,
                    "action_grads": action_grads, 
                    "final_reward": reward,
                    "values": values,
                    "logits": logits,
                    }
        all_features.append(features)

    return all_features


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