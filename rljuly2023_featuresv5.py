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


FEATS_DIR = '/workspace/manoj/rljuly2023_featuresv6'
print("jacobians of everything ....")
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

    model_dirpath = '/'.join(model_filepath.split('/')[:-1])
    is_poisoned = load_ground_truth(model_dirpath)
    print("ground truth is poisoned ",is_poisoned)

    model.to(device)
    model.eval()

    env_string = "MiniGrid-LavaCrossingS9N1-v0"
    logging.info('Evaluating on {}'.format(env_string))
    # Run episodes through an environment to collect what may be relevant information to trojan detection
    all_features =  get_jacobian_features(env_string, model)
    return all_features
  

if __name__ == '__main__':

    for model_idx in range(238):
        model_name =  'id-%08d' % model_idx
        model_filepath = os.path.join(root(),'models', model_name , 'model.pt')
        all_features = infer(model_filepath)
        feat_path = os.path.join(FEATS_DIR, f'{model_name}.pt')
        torch.save(all_features, feat_path)
        print(f"Saved to {feat_path}")
