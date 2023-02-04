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

### TODOs ###
# TODO make sure the model loads correctly. might need to find that other file that has the architecture *done*
# TODO identify the right mapping between acoustics and waveforms *done*
# TODO add spectrogram conversion to dataset *done*
# TODO figure out the way to adjust the input in the direction of the gradient

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

    def train(self, dataloader):
        self.model.train()
        for i, (x, y) in enumerate(dataloader):
            breakpoint()
            x = x.to(self.device) 
            x.requires_grad = True  # need input grad for adversarial  
            y = y.to(self.device)
            y_hat = self.model(x)
            loss = self.loss_fn(y_hat, y)
            self.optimizer.zero_grad()
            loss.backward()
            # adjust the input in the direction of the gradient
            self.optimizer.step()
            if i % 10 == 0:
                print(f'Epoch {i}: {loss.item()}')
        return 0


def main():
    dataset = data.ToyDataset()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    # load model from config.model_path
    model = AcousticEstimator()
    model_ckpt = torch.load(config.model_path)
    model.load_state_dict(model_ckpt['model_state_dict'])
    # create Cradle 
    cradle = Cradle(model)
    cradle.train(dataloader)
    return 0 

if __name__ == '__main__':
    main()