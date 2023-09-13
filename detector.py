# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.


import logging
import os
import json

import jsonpickle
import pickle
import numpy as np
import torch
from utils.abstract import AbstractDetector
from utils.models import load_model, load_models_dirpath, load_ground_truth

from utils import trojai_utils
from utils.trojai_utils import *
from utils.rljuly2023_utils import *

import torch
import torch_ac
import gym
from gym_minigrid.wrappers import ImgObsWrapper
from scipy import stats
from types import SimpleNamespace


def get_features(features, feature_name):
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
    return [{"feats": torch.FloatTensor(feature_flat), "reward": final_reward}]


def predict(saved_config, raw_feat):
    scores = []
    config = SimpleNamespace()
    config.raw_size = 441 + 1 # for value
    config.feat_size = 30
    config.nlayers_1 = 1
    config.hidden_size = 60
    config.feature_name = "action_grads_value"
    with torch.no_grad():
        kfold_state_dicts = saved_config.pop("state_dicts")
        for fold_state in kfold_state_dicts:
            ## remaining is config params
            model = NLP2023MetaNetwork2(raw_size = config.raw_size, feat_size=config.feat_size,
                                         hidden_size=config.hidden_size, nlayers_1=config.nlayers_1)
            model.load_state_dict(fold_state['state_dict'])
            score = model(raw_feat).squeeze().item()
            scores.append(score)
    print(scores)
    # majority voting
    trojan_probability = sum(scores) / len(scores)
    # trojan_probability = stats.mode(scores).mode
    return float(trojan_probability)


class Detector(AbstractDetector):
    def __init__(self, metaparameter_filepath, learned_parameters_dirpath):
        """Detector initialization function.

        Args:
            metaparameter_filepath: str - File path to the metaparameters file.
            learned_parameters_dirpath: str - Path to the learned parameters directory.
        """
        metaparameters = json.load(open(metaparameter_filepath, "r"))

        self.metaparameter_filepath = metaparameter_filepath
        self.learned_parameters_dirpath = learned_parameters_dirpath
        self.model_filepath = os.path.join(self.learned_parameters_dirpath, "model.bin")
        self.models_padding_dict_filepath = os.path.join(self.learned_parameters_dirpath, "models_padding_dict.bin")
        self.model_layer_map_filepath = os.path.join(self.learned_parameters_dirpath, "model_layer_map.bin")
        self.layer_transform_filepath = os.path.join(self.learned_parameters_dirpath, "layer_transform.bin")

    def automatic_configure(self, models_dirpath: str):
        """Configuration of the detector iterating on some of the parameters from the
        metaparameter file, performing a grid search type approach to optimize these
        parameters.

        Args:
            models_dirpath: str - Path to the list of model to use for training
        """

        # dataset=trinity.extract_dataset(models_dirpath,ts_engine=trinity.ts_engine,params=self.params)
        # splits=crossval.split(dataset,self.params)
        # ensemble,perf=crossval.train(splits,self.params)
        # torch.save(ensemble,os.path.join(self.learned_parameters_dirpath,'model.pt'))
        # return True

        for random_seed in np.random.randint(1000, 9999, 10):
            pass
        return True

    def manual_configure(self, models_dirpath: str):
        return self.automatic_configure(models_dirpath)


    def infer(
            self,
            model_filepath,
            result_filepath,
            scratch_dirpath,
            examples_dirpath,
            round_training_dataset_dirpath,
    ):
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



        fvs = get_features(all_features,"action_grads_value")

        if self.learned_parameters_dirpath is not None:
            try:
                ensemble = torch.load(os.path.join(self.learned_parameters_dirpath, 'bestrljuly2023_modelv5_jacobian.pth'))
            except:
                ensemble = torch.load(os.path.join('/', self.learned_parameters_dirpath, 'bestrljuly2023_modelv5_jacobian.pth'))

            probability = predict(ensemble,fvs)
        else:
            probability = 0.5


        # clip the probability to reasonable values
        probability = np.clip(probability, a_min=0.01, a_max=0.99)

        # Test scratch space
        with open(os.path.join(scratch_dirpath, 'test.txt'), 'a+') as fh:
            fh.write(model_filepath + "," + str(probability))

        # write the trojan probability to the output file
        with open(result_filepath, "w") as fp:
            s = model_filepath
            fp.write(str(probability))

        logging.info("Trojan probability: %f", probability)
        return probability

