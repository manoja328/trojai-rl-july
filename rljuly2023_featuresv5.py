import numpy as np
import logging
import csv
import torch
import os
import os, random
from tqdm import tqdm
from utils.rljuly2023_utils import *
from utils.trojai_utils import *
from utils.models import load_model, load_ground_truth

import torch
import torch_ac
import gym
from gym_minigrid.wrappers import ImgObsWrapper
import torch.autograd.functional as AF

FEATS_DIR = '/workspace/manoj/rljuly2023_featuresv6'
print("jacobians of everything ....")
os.makedirs(FEATS_DIR,exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def infer(model_filepath):
    """Method to predict whether a model is poisoned (1) or clean (0).

    Args:
        model_filepath:
        result_filepath:
        scratch_dirpath:
        examples_dirpath:
        round_training_dataset_dirpath:
        tokenizer_filepath:
    """

    # load the model
    model, model_repr, model_class = load_model(model_filepath)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info("Using compute device: {}".format(device))

    model_dirpath = '/'.join(model_filepath.split('/')[:-1])
    is_poisoned = load_ground_truth(model_dirpath)
    print("ground truth is poisoned ",is_poisoned)

    model.to(device)
    model.eval()

    preprocess = torch_ac.format.default_preprocess_obss

    env_string = "MiniGrid-LavaCrossingS9N1-v0"
    logging.info('Evaluating on {}'.format(env_string))

    # Run episodes through an environment to collect what may be relevant information to trojan detection
    # Construct environment and put it inside a observation wrapper
    env = ImgObsWrapper(gym.make(env_string))

    episodes = 1
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
        while not done:
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
                    "logits": dist.logits,
                    "ground_truth": is_poisoned,
                    }
        all_features.append(features)

    # import pdb; pdb.set_trace()
    return all_features


if __name__ == '__main__':

    for model_idx in range(238):
        model_name =  'id-%08d' % model_idx
        model_filepath = os.path.join(root(),'models', model_name , 'model.pt')
        all_features = infer(model_filepath)
        feat_path = os.path.join(FEATS_DIR, f'{model_name}.pt')
        torch.save(all_features, feat_path)
        print(f"Saved to {feat_path}")
