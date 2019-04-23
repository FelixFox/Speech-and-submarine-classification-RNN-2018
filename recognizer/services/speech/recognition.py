import os
from scipy.io import wavfile
from scipy import signal
import numpy as np
import math
import keras
from keras.models import load_model
import keras.backend as K
import tensorflow as tf


from .preprocessing import SpeechAudioFilePreprocessor


model_file = os.path.join(os.path.dirname(
    __file__), 'nnmodel', 'cnn_no_mp_lstm_a94.h5')
word_classes_file = os.path.join(os.path.dirname(
    __file__), 'preprocDataset', 'word_classes.npy')
recorded_file = os.path.join(os.path.dirname(__file__), 'output.wav')

model = load_model(model_file)
graph = tf.get_default_graph()


def with_default_graph(func):
    def wrapper(self=None):
        global graph
        with graph.as_default():
            return func(self)

    return wrapper


class SpeechRecognizer:
    def __init__(self, model_file=model_file,
                 word_classes_file=word_classes_file,
                 recorded_file=recorded_file):
        global model
        self.f_shape = model.layers[0].input_shape
        self.word_classes = np.load(word_classes_file)
        self.recorded_file = recorded_file
        self.audio_preprocessor = SpeechAudioFilePreprocessor(
            self.recorded_file)

    def wav_to_features(self):
        return self.audio_preprocessor.wav_to_mfcc_spec()

    @with_default_graph
    def predict(self):
        mels = self.wav_to_features()
        mels = np.expand_dims(mels, axis=2)
        mels = np.expand_dims(mels, axis=0)
        global model
        y = model.predict(mels)
        predicted_class = str(self.word_classes[y.argmax()])
        score = y.max()
        y = np.reshape(y, len(y[0]))
        y_dict = dict(zip(self.word_classes, y))
        return {'score': score, 'predicted_class': predicted_class, 'y': y_dict}

    def dispose(self):
        
        del self.word_classes
        del self.recorded_file


if __name__ == '__main__':
    sr = SpeechRecognizer()
    print(sr.f_shape)
    result = sr.predict()
    print(result)
