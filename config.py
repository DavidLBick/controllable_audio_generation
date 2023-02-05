import numpy as np

lr = 1e-3
batch_size = 1
model_path = "./hamming_lld_estimator_13mse_13mae.pt"
waveform_dir = "/media/konan/DataDrive/temp/toy-acoustic/waveform/clean/"
acoustic_dir = "/media/konan/DataDrive/temp/toy-acoustic/acoustic/clean/"
acoustic_features = [
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