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
        self.loss_fn = torch.nn.MSELoss()
        self.model = model.to(self.device)
        self.patience = 20  # TODO adjust 
        self.cost_threshold = 0.1  # TODO adjust 
        self.tolerance = 0.01  # TODO adjust

    def vary_one_feature(self, spec, feature_idx, multiplier):
        lr = 1
        acoustic = self.model(spec)  # TODO seems like they are normalized, but should confirm
        feat_std = torch.std(acoustic[:, :, feature_idx])
        varied_acstc = acoustic.clone().to(self.device)
        varied_acstc[:, :, feature_idx] += multiplier * feat_std  # TODO might want to use something else for std
        cost = self.loss_fn(acoustic, varied_acstc)
        grad = torch.autograd.grad(cost, spec,
                                   retain_graph=False, create_graph=False)[0]  # credit to https://github.com/Harry24k/adversarial-attacks-pytorch/blob/master/torchattacks/attacks/fgsm.py
        step = 0
        repeats = 0
        while abs(cost) > self.cost_threshold:
            print(cost, step)
            #breakpoint()
            spec = spec - lr * grad.sign()
            acstc_i = self.model(spec)
            prev_cost = cost.detach().clone()
            cost = self.loss_fn(acstc_i, varied_acstc)
            if abs(cost - prev_cost) < self.tolerance:
                if repeats > self.patience:
                    lr = lr / 2
                    # clamp lr to min 0.001
                    if lr < 0.1:
                        lr = 0.1
                    repeats = 0
                else:
                    repeats += 1
            grad = torch.autograd.grad(cost, spec,
                                        retain_graph=False, create_graph=False)[0]
            step += 1
        adv_spec = spec.detach()
        return adv_spec

    def vary_one_feature_optimizer(self, spec, feature_idx, multiplier):
        optim = torch.optim.Adam([spec], lr=0.75)
        acoustic = self.model(spec)  # TODO seems like they are normalized, but should confirm
        feat_std = torch.std(acoustic[:, :, feature_idx])
        varied_acstc = acoustic.clone().to(self.device)
        varied_acstc[:, :, feature_idx] += multiplier * feat_std
        cost = self.loss_fn(acoustic, varied_acstc)
        grad = torch.autograd.grad(cost, spec,
                                   retain_graph=False, create_graph=False)[0]  # credit to https://github.com/Harry24k/adversarial-attacks-pytorch/blob/master/torchattacks/attacks/fgsm.py
        step = 0
        optim.step()    
        optim = torch.optim.Adam([spec], lr=0.75)
        while abs(cost) > self.cost_threshold:
            print(cost, step)
            acstc_i = self.model(spec)
            prev_cost = cost.clone()
            cost = self.loss_fn(acstc_i, varied_acstc)
            grad = torch.autograd.grad(cost, spec, retain_graph=False, create_graph=False)[0]
            optim.step()
            optim = torch.optim.Adam([spec], lr=0.75)
            step += 1
        adv_spec = spec.detach()
        return adv_spec

    def cntrl_gen(self, dataloader):
        self.model.train()
        gen_features = dict()
        for feature_idx in range(0, len(config.acoustic_features)):
            feature = config.acoustic_features[feature_idx]
            gen_features[feature] = []
            for i, (spec, gt_acstc) in tqdm(enumerate(dataloader)):
                if i > 3:
                    break
                spec = spec.to(self.device) 
                spec.requires_grad = True  # need input grad for adversarial  
                gt_acstc = gt_acstc.to(self.device)  # TODO why is # of time steps different for gt_acstc and estimated
                spec_atcks = []
                """for multiplier in range(-5, 5):
                    gen_audio = self.vary_one_feature_optimizer(spec, feature_idx, multiplier)
                    spec_atcks.append(gen_audio)"""
                gen_audio = self.vary_one_feature(spec, feature_idx, 5)
                gen_features[feature].append(spec_atcks)
        return 0

def main():
    dataset = data.ToyDataset()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=False)
    model = AcousticEstimator()
    model_ckpt = torch.load(config.model_path)
    model.load_state_dict(model_ckpt['model_state_dict'])
    cradle = Cradle(model)
    cradle.cntrl_gen(dataloader)
    return 0 

if __name__ == '__main__':
    main()