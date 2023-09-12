import numpy as np
import logging
import csv
import torch
import os
import pandas as pd
import os, random
from tqdm import tqdm
from utils.rljuly2023_utils import *
from utils.trojai_utils import *
from utils.models import load_model

FEATS_DIR = '/workspace/manoj/rljuly2023_featuresv2'
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

    # Utilize open source minigrid environment model was trained on
    env_string_filepath = os.path.join(examples_dirpath, 'env-string.txt')
    with open(env_string_filepath) as env_string_file:
        env_string = env_string_file.readline().strip()
    logging.info('Evaluating on {}'.format(env_string))



    # Run episodes through an environment to collect what may be relevant information to trojan detection
    # Construct environment and put it inside a observation wrapper
    env = ImgObsWrapper(gym.make(env_string))

    episodes = 100
    all_features = []
    final_rewards = []
    with torch.no_grad():
        # Episode loop
        for _ in range(episodes):
            # Reset environment after episode and get initial observation
            obs = env.reset()
            done = False
            # Per episode loop
            features = []
            while not done:
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

            features.append(reward)
            all_features.append(features)

            # Collect episode performance data (just the last reward of the episode)
            final_rewards.append(reward)


    all_features =  np.array(all_features)
    print("feature shape ",all_features.shape)
    return all_features


if __name__ == '__main__':
    meta_df = get_metadata()
    for model_idx, row in meta_df.iterrows():
        model_filepath = os.path.join(root(),'models', row.model_name, 'model.pt')
        eng = RL2023july(model_filepath,"cpu")
        # eng = load_engine(model_idx, device)
        print(eng.root)
        print("is trojaned:", eng.istrojaned())
        #weight analysis
        all_features = infer(model_filepath)
        feat_path = os.path.join(FEATS_DIR, f'{row.model_name}.pt')
        torch.save(all_features, feat_path)
        print(f"Saved to {feat_path}")
