Using our acoustic-parameter estimator from ICASSP follow-up to InterSpeech work to perform controllable audio generation. Idea is similar to adversarial attacks, where we can backpropagate the difference from target acoustic features to audio, to create an audio with those features. After inverting STFT we should be able to identify the perceptual effect of changing a specific set of acoustic features in the audio. 

InterSpeech paper: https://www.isca-speech.org/archive/pdfs/interspeech_2022/yang22x_interspeech.pdf
ICASSP paper: TBA
