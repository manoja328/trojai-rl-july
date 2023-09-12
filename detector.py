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
from utils.models import load_model, load_models_dirpath

from utils import trojai_utils
from utils.trojai_utils import *

import torch
import torch_ac
import gym
from gym_minigrid.wrappers import ImgObsWrapper
from scipy import stats


def predict(saved_config, raw_feat):
    scores = []
    with torch.no_grad():
        kfold_state_dicts = saved_config.pop("state_dicts")
        for fold_state in kfold_state_dicts:
            ## remaining is config params
            model = NLP2023MetaNetwork(raw_size=100, feat_size=60, hidden_size=30, nlayers_1=1)
            model.load_state_dict(fold_state['state_dict'])
            score = model(raw_feat).squeeze().item()
            scores.append(score)
    # majority voting
    # trojan_probability = sum(scores) / len(scores)
    print(scores)
    trojan_probability = stats.mode(scores).mode
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


        all_features =  np.array(all_features)
        print("feature shape ",all_features.shape)
        fvs = [{"feats": torch.FloatTensor(all_features)}]

        if self.learned_parameters_dirpath is not None:
            try:
                ensemble = torch.load(os.path.join(self.learned_parameters_dirpath, 'bestrljuly2023_model.pth'))
            except:
                ensemble = torch.load(os.path.join('/', self.learned_parameters_dirpath, 'bestrljuly2023_model.pth'))

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

