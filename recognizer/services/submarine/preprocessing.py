import os
import wave
import glob
import sklearn
import scipy
from scipy.io import wavfile
from scipy.fftpack import fft
from scipy import signal
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


filename = os.path.join(os.path.dirname(__file__), 'output.wav')


class AudioFilePreprocessor:
    def __init__(self, filename=filename,
                 resample_rate=1024,
                 resample_mode=True,
                 normalize_mode=True,
                 max_freq=130,
                 f_step=1,
                 prep_channel=1):

        self.filename = filename
        self.resample_rate = resample_rate
        self.resample_mode = resample_mode
        self.normalize_mode = normalize_mode
        self.max_freq = max_freq
        self.f_step = f_step
        self.prep_channel = prep_channel

    def resample_audio(self):
        sample_rate, samples = wavfile.read(self.filename)
        resampled = signal.resample(samples, int(
            self.resample_rate/sample_rate * samples.shape[0]))
        return self.resample_rate, resampled.astype(np.int16)

    def get_samplerate(self):
        with wave.open(self.filename, "rb") as wav_file:
            return wav_file.getframerate()

    def get_fft(self):

        if (self.resample_mode):
            sample_rate, samples = self.resample_audio()

        else:
            sample_rate, samples = wavfile.read(self.filename)

        T = 1.0 / sample_rate
        if (self.prep_channel in [1, 2, 3, 4] and self.prep_channel <= samples.shape[1]):
            samples = samples[:, self.prep_channel - 1]
        N = samples.shape[0]
        fft_result = fft(samples)
        fft_Freqs = np.linspace(0.0, 1.0/(2.0*T), N//2)
        # FFT is simmetrical, so we take just the first half
        # FFT is also complex, to we take just the real part (abs)
        fft_Amps = (2.0/N) * np.abs(fft_result[0:N//2])
        fft_Amps = np.expand_dims(fft_Amps, axis=1)
        # normalizing
        if self.normalize_mode:

            scaler = MinMaxScaler()
            fft_Amps = scaler.fit_transform(fft_Amps)

        # cut data relatet to unnecessary frequencies
        if self.max_freq:
            fft_Freqs, fft_Amps = self.reduce_to_max_freq_fft(
                fft_Freqs, fft_Amps)

        return fft_Freqs, fft_Amps

    def get_fixed_sized_fft(self, samples, size):
        if len(samples) > size:
            samples = samples[:size]

        # append zeros
        while len(samples) < size:
            samples = np.append(samples, 0)

        return samples

    def reduce_to_max_freq_fft(self, freqs, amps):
        freqs_r = np.around(freqs)
        index = np.where(freqs_r == self.max_freq)[0][0]
        fft_freqs = freqs[:index]
        fft_amps = amps[:index]
        return fft_freqs, fft_amps

    def reduce_by_step_fft(self, freqs, amps):
        freqs = freqs[::self.f_step]
        amps = amps[::self.f_step]
        return freqs, amps



def PlotFFTFromFile(filename=filename, x_size=14, y_size=8):
    rec = AudioFilePreprocessor()
    
    fig = plt.figure(figsize=(x_size, y_size))
    ax_ = fig.add_subplot(212)
    ax_.set_title('FFT of  ' + filename)
    ax_.set_ylabel('Amplitude')
    ax_.set_xlabel('Frequency')
    xf_, vals_ = rec.get_fft()
    
    ax_.grid()
    ax_.plot(xf_, vals_)
    # fig.show()
    # plt.show()
    return fig

if __name__ == "__main__":
    prep = AudioFilePreprocessor()
    fft, amps = prep.get_fft()
    PlotFFTFromFile()
