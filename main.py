import torch
import torchaudio
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
import glob
import gc
from tqdm import tqdm
import matplotlib.pyplot as plt
import config
import data
import copy

class AcousticEstimator(torch.nn.Module):
    def __init__(self):
        super(AcousticEstimator, self).__init__()
        self.lstm = torch.nn.LSTM(642, 256, 4, bidirectional=True, batch_first=True)
        self.linear1 = torch.nn.Linear(512, 256)
        self.linear2 = torch.nn.Linear(256, 128)
        self.linear3 = torch.nn.Linear(128, 25)
        self.act = torch.nn.ReLU()
        
    def forward(self, A0):
        A1, _ = self.lstm(A0)
        Z1 = self.linear1(A1)
        A2 = self.act(Z1)
        Z2 = self.linear2(A2)
        out = self.act(Z2)
        out = self.linear3(out)
        return out 

class Cradle(torch.nn.Module):
    def __init__(self, model):
        super(Cradle, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        self.loss_fn = torch.nn.MSELoss()
        self.model = model.to(self.device)

    def vary_one_feature(self, audio, acoustic, feature_idx, multiplier):
        # TODO might want to use something else for std
        feat_std = torch.std(acoustic[:, :, feature_idx])
        varied_acstc = acoustic.clone().to(self.device)
        varied_acstc[:, :, feature_idx] += multiplier * feat_std
        breakpoint()
        cost = -self.loss_fn(acoustic, varied_acstc)
        grad = torch.autograd.grad(cost, audio,
                                   retain_graph=False, create_graph=False)[0]  # credit to https://github.com/Harry24k/adversarial-attacks-pytorch/blob/master/torchattacks/attacks/fgsm.py

        # TODO might have to play with iterative or other methods                                    
        adv_audio = audio + self.eps * grad.sign()
        return adv_audio

    def cntrl_gen(self, dataloader):
        self.model.train()
        gen_features = dict()
        for feature_idx in range(0, len(config.acoustic_features)):
            feature = config.acoustic_features[feature_idx]
            gen_features[feature] = []
            for i, (spec, gt_acstc) in enumerate(dataloader):
                if i > 3:
                    break
                spec = spec.to(self.device) 
                spec.requires_grad = True  # need input grad for adversarial  
                gt_acstc = gt_acstc.to(self.device)  # TODO why is # of time steps different for gt_acstc and estimated
                spec_atcks = []
                estimated_acstc = self.model(spec)  # TODO seems like they are normalized, but should confirm
                for multiplier in range(-5, 5):
                    gen_audio = self.vary_one_feature(spec, estimated_acstc, feature_idx, multiplier)
                    spec_atcks.append(gen_audio)
                gen_features[feature].append(spec_atcks)
        return 0

def main():
    dataset = data.ToyDataset()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    model = AcousticEstimator()
    model_ckpt = torch.load(config.model_path)
    model.load_state_dict(model_ckpt['model_state_dict'])
    cradle = Cradle(model)
    cradle.cntrl_gen(dataloader)
    return 0 

if __name__ == '__main__':
    main()