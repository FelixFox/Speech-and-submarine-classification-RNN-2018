import numpy as np
import librosa
from math import ceil
import os
import wave
import librosa.display

filename = os.path.join(os.path.dirname(__file__), 'output.wav')
resample_dim = 3
hop_length = 256
n_fft = 1024
n_mels = 128
sample_r = 16000
# n_parts = np.ceil(sample_r/256) #parts of time sequence


class SpeechAudioFilePreprocessor:
    def __init__(self, filename=filename):
        self.filename = filename
        self.n_mels = 128
        self.hop_length = 256
        self.n_fft = 1024

    def wav_to_mfcc(self,  max_pad_len=11):
        wave, sample_rate = librosa.load(self.filename, mono=True, sr=None)
        wave = wave[::resample_dim]
        mfcc = librosa.feature.mfcc(wave, sr=16000, n_mfcc=self.n_mels)
        pad_width = max_pad_len-mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=(
            (0, 0), (0, pad_width)), mode='constant')
        return mfcc

    def wav_to_mfcc_spec(self):
        wave, sample_rate = librosa.load(self.filename, mono=True, sr=None)
        wave = wave[:sample_rate]
        pad_width = sample_rate-len(wave)
        wave = np.pad(wave, (0, pad_width), 'constant')
        M_coef = librosa.feature.melspectrogram(
            y=wave, sr=sample_rate, hop_length=self.hop_length, n_fft=self.n_fft, n_mels=self.n_mels)
        M_coef_db = librosa.power_to_db(M_coef)
        return M_coef_db.T

    def get_samplerate(self):
        with wave.open(self.filename, "rb") as wav_file:
            return wav_file.getframerate()


if __name__ == "__main__":

    
    prep = SpeechAudioFilePreprocessor(filename)
    mels = prep.wav_to_mfcc_spec()
    
    plt.figure(figsize=(10, 4))
    
    librosa.display.specshow(mels)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel spectrogram')
    plt.tight_layout()
    
    print(type(mels))
    print(type(mels.tolist()))
    print(mels.shape)
    print(len(mels.tolist()))
    
