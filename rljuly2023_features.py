import numpy as np
import csv
import torch
import os
import pandas as pd
import os, random
from tqdm import tqdm
from utils.rljuly2023_utils import *
from utils.trojai_utils import *
from utils.models import load_model

FEATS_DIR = '/workspace/manoj/rljuly2023_features'
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

    all_features = extract_params_hist(model)  # encoder only and no embeddings
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
