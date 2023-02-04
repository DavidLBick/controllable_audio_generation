import torch
import torchaudio
from torchsummaryX import summary

import pandas as pd
import numpy as np

import random
from sklearn.model_selection import train_test_split

import glob
import gc
from tqdm import tqdm

import matplotlib.pyplot as plt

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
feature_names = [
    'Loudness_sma3',
    'alphaRatio_sma3',
    'hammarbergIndex_sma3',
    'slope0-500_sma3',
    'slope500-1500_sma3',
    'spectralFlux_sma3',
    'mfcc1_sma3',
    'mfcc2_sma3',
    'mfcc3_sma3',
    'mfcc4_sma3',
    'F0semitoneFrom27.5Hz_sma3nz',
    'jitterLocal_sma3nz',
    'shimmerLocaldB_sma3nz',
    'HNRdBACF_sma3nz',
    'logRelF0-H1-H2_sma3nz',
    'logRelF0-H1-A3_sma3nz',
    'F1frequency_sma3nz',
    'F1bandwidth_sma3nz',
    'F1amplitudeLogRelF0_sma3nz',
    'F2frequency_sma3nz',
    'F2bandwidth_sma3nz',
    'F2amplitudeLogRelF0_sma3nz',
    'F3frequency_sma3nz',
    'F3bandwidth_sma3nz',
    'F3amplitudeLogRelF0_feature_names']
    
feature_names = ['CLEAN_' + feature for feature in feature_names] + ['NOISY_' + feature for feature in feature_names]
feature_names

clean_acoustic_paths = sorted(glob.glob('/media/konan/DataDrive/temp/toy-acoustic/acoustic/clean/*/*.npy'))
noisy_acoustic_paths = sorted(glob.glob('/media/konan/DataDrive/temp/toy-acoustic/acoustic/noisy/*/*.npy'))


evaluation_metrics_paths = '/media/konan/DataDrive/temp/toy-acoustic/eval_metrics/metrics_noisy(toy).csv'

total_nums = len(clean_acoustic_paths)
print(total_nums)

def get_relative_acoustic(clean_acoustic_paths, noisy_acoustic_paths):
    
    mu = np.array(
            [ 2.31615782e-01, -5.02114248e+00,  7.16793156e+00,  1.40047576e-02,
             -1.44424592e-03,  1.18291244e-01,  7.16937304e+00,  5.01161051e+00,
              7.38044071e+00,  1.30544746e+00,  7.16783571e+00,  7.72617990e-03,
              3.78611624e-01,  1.80594587e+00,  2.74223471e+00,  7.16790104e+00,
              2.29371735e+02,  2.61031281e+02, -2.86713428e+01,  4.58741486e+02,
              2.72984955e+02, -2.86713428e+01,  4.58874390e+02,  2.71175812e+02,
             -2.86713428e+01], dtype=np.float32)
    std = np.array(
            [ 4.24716711e-01, 1.09750290e+01, 1.51086359e+01, 2.98775751e-02,
              1.85245797e-02, 2.39421308e-01, 1.63376312e+01, 1.22261524e+01,
              1.53735695e+01, 1.42613926e+01, 1.21981163e+01, 2.58955006e-02,
              8.05543840e-01, 3.83967781e+00, 6.79308844e+00, 1.41308403e+01,
              3.49271667e+02, 6.28384338e+02, 6.05799637e+01, 6.89079407e+02,
              5.62089905e+02, 6.05799637e+01, 1.09140088e+03, 5.42341919e+02,
              6.05799637e+01], dtype=np.float32)
    
    clean_acoustic = torch.tensor(np.asarray([np.load(p) for p in tqdm(clean_acoustic_paths)]))
    noisy_acoustic = torch.tensor(np.asarray([np.load(p) for p in tqdm(noisy_acoustic_paths)]))
    
    # Normalization
    clean_acoustic = (clean_acoustic - mu) / std
    noisy_acoustic = (noisy_acoustic - mu) / std 
    
    # Because PESQ and STOI are relative evaluation metrics, the input must include both clean and noisy(enhanced) acoustic.
    acoustic = torch.cat((clean_acoustic, noisy_acoustic), 2)
    
    return acoustic

class ToyDataset(torch.utils.data.Dataset):

    def __init__(self):
        
        self.acoustic              = get_relative_acoustic(clean_acoustic_paths, noisy_acoustic_paths)
        self.evaluation_metrics    = np.asarray(pd.read_csv(evaluation_metrics_paths).loc[:, ['pesq', 'stoi']])
        self.evaluation_metrics    = torch.tensor(self.evaluation_metrics)
        
        assert len(self.evaluation_metrics) == len(self.acoustic)
        
        self.length = len(self.evaluation_metrics)
        

        
    def __len__(self):

        return self.length

    def __getitem__(self, i):

        x = self.acoustic[i]
        y = self.evaluation_metrics[i]
        
        return x, y



