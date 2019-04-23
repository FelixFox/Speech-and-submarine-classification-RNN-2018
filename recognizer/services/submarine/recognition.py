from scipy.io import wavfile
from scipy import signal
import numpy as np
import math
import keras
from keras.models import load_model
from .preprocessing import AudioFilePreprocessor
import keras.backend as K
import gc
import os
import tensorflow as tf

model_file = os.path.join(os.path.dirname(__file__), 'nnmodel', 'cnn.h5')
word_classes_file = os.path.join(os.path.dirname(
    __file__), 'preprocDataset', 'sub_classes_(65,39368).npy')
recorded_file = os.path.join(os.path.dirname(__file__), 'output.wav')

model = load_model(model_file)
graph = tf.get_default_graph()

def with_default_graph(func):
    def wrapper(self=None):
        global graph
        with graph.as_default():
            return func(self)
        
    return wrapper

class SubmarineRecognizer:
    def __init__(self, model_file=model_file,
                 word_classes_file=word_classes_file,
                 recorded_file=recorded_file):
        global model
        self.f_shape = model.layers[0].input_shape
        self.word_classes = np.load(word_classes_file)
        self.recorded_file = recorded_file
        self.audio_preprocessor = AudioFilePreprocessor(self.recorded_file)
        

    def wav_to_features(self):
        freqs, amps = self.audio_preprocessor.get_fft()
        amps = self.audio_preprocessor.get_fixed_sized_fft(
            amps.T[0], self.f_shape[2])
        return freqs, amps

    @with_default_graph
    def predict(self):
        freqs, amps = self.wav_to_features()
        amps = np.expand_dims(amps, axis=0)
        amps = np.expand_dims(amps, axis=0)
        global model
        y = model.predict(amps)
        predicted_class = str(self.word_classes[y.argmax()])
        score = y.max()
        y = np.reshape(y, len(y[0][0]))
        y_dict = dict(zip(self.word_classes, y))
        return {'score': score, 'predicted_class': predicted_class, 'y': y_dict}

    def dispose(self):
        
        del self.word_classes
        del self.recorded_file


if __name__ == '__main__':
    rec = SubmarineRecognizer()
    result = rec.predict()
    print(result)
