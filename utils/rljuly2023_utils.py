import numpy as np
import csv
import os
import copy
import json
import torch.nn as nn
import torch.nn.functional as F
import torch

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