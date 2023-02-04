import config 
import numpy as np 
import pandas as pd 
import torch
import torchaudio
import glob
import tqdm

class ToyDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.acoustics = self.get_acoustics()
        # TODO load waveforms 
        self.waveforms = self.get_waveforms()
        self.nfft = 640
        self.hop_length = 160

    def get_acoustics(self):
        mu, std = config.mu, config.std
        clean_acoustic_paths = sorted(glob.glob(f'{config.acoustic_dir}/*/*.npy'))
        clean_acoustics = torch.tensor(np.asarray([np.load(p) for p in tqdm.tqdm(clean_acoustic_paths)]))
        # Normalization
        clean_acoustics = (clean_acoustics - mu) / std
        return clean_acoustics

    def get_waveforms(self):
        clean_waveform_paths = sorted(glob.glob(f'{config.waveform_dir}/*/*.wav'))
        # TODO depending on how big this is, might need to just store the paths and load them in the getitem function
        # read in each waveform using torchaudio and store it in a list
        wavs = []
        for p in tqdm.tqdm(clean_waveform_paths):
            wav, sr = torchaudio.load(p)
            wavs.append(wav)
        return wavs

    def get_stft(self, wav, return_short_time_energy = False):
        spec = torch.stft(wav, n_fft=self.nfft, hop_length=self.hop_length, return_complex=False)
        spec_real = spec[..., 0]
        spec_imag = spec[..., 1]  
        spec = spec.permute(0, 2, 1, 3).reshape(spec.size(dim=0), -1, 642)
        spec = spec.squeeze()
        if return_short_time_energy:
            st_energy = torch.mul(torch.sum(spec_real**2 + spec_imag**2, dim = 1), 2/self.nfft)
            assert spec.size(dim=1) == st_energy.size(dim=1)
            return spec.float(), st_energy.float()
        else: 
            return spec.float()

    def get_st_energy(self, spec):                               
        spec_real = spec[..., :spec.size(dim=-1)//2]
        spec_imag = spec[..., spec.size(dim=-1)//2:]                                         
        """
            spec_real ==> (B, T, 257)
            spec_imag ==> (B, T, 257)
        """
        st_energy = torch.mul(torch.sum(spec_real**2 + spec_imag**2, dim = 2), 2/spec.size(dim=-1))
        assert spec.size(dim=1) == st_energy.size(dim=1)
        return st_energy.float()     
        
    def __len__(self):
        return len(self.acoustics)

    def __getitem__(self, i):
        wav = self.waveforms[i]
        x = self.get_stft(wav)
        y = self.acoustics[i]
        return x, y