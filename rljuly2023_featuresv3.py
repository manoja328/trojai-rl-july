import numpy as np
import logging
import csv
import torch
import os
import os, random
from tqdm import tqdm
from utils.rljuly2023_utils import *
from utils.trojai_utils import *
from utils.models import load_model

import torch
import torch_ac
import gym
from gym_minigrid.wrappers import ImgObsWrapper


FEATS_DIR = '/workspace/manoj/rljuly2023_featuresv3'
os.makedirs(FEATS_DIR,exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    model.to(device)
    model.eval()

    preprocess = torch_ac.format.default_preprocess_obss


    env_string = "MiniGrid-LavaCrossingS9N1-v0"
    logging.info('Evaluating on {}'.format(env_string))



    # Run episodes through an environment to collect what may be relevant information to trojan detection
    # Construct environment and put it inside a observation wrapper
    env = ImgObsWrapper(gym.make(env_string))

    episodes = 100
    all_features = []
    final_rewards = []
    max_episode_length = 100
    with torch.no_grad():
        # Episode loop
        for _ in range(episodes):
            # Reset environment after episode and get initial observation
            obs = env.reset()
            done = False
            # Per episode loop
            features = []
            while not done:
                ##add the trigger
                obs = ((obs + 10) % 256).astype(np.float)
                # Preprocessing function to prepare observation from env to be given to the model
                obs = preprocess([obs], device=device)
                # Use env observation to get action distribution
                dist, value = model(obs)
                # Sample from distribution to determine which action to take
                action = dist.sample()
                action = action.cpu().detach().numpy()
                # Use action to step environment and get new observation
                obs, reward, done, info = env.step(action)
                features.append(value.cpu().detach().item())

            features.extend([0]*(max_episode_length- len(features)))
            features[-1] = reward
            all_features.append(features[:max_episode_length])

    # import pdb; pdb.set_trace()
    all_features =  np.array(all_features)
    print("feature shape ",all_features.shape)
    return all_features


if __name__ == '__main__':


    for model_idx in range(238):
        model_name =  'id-%08d' % model_idx
        model_filepath = os.path.join(root(),'models', model_name, 'model.pt')
        all_features = infer(model_filepath)
        feat_path = os.path.join(FEATS_DIR, f'{model_name}.pt')
        torch.save(all_features, feat_path)
        print(f"Saved to {feat_path}")
