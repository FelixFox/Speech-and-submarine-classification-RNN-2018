import keras
from keras.models import load_model
from keras.datasets import mnist
from keras.layers import Input, Dense, GlobalAveragePooling1D, Reshape, LSTM, TimeDistributed, BatchNormalization, Activation, Dropout, Conv2D, Conv1D, MaxPooling1D, MaxPooling2D, Flatten, Concatenate, Add, ZeroPadding2D, AveragePooling2D
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.regularizers import l2
import numpy as np

batch_size = 1024
num_classes = 16
epochs = 30
train_num = 28000
val_num = 5000
ds_path = './DatasetMFCC/'

time = 63
input_shape = 128
w = 16
h = 8
x = np.load(ds_path+'X_MFCC_63_128.npy')
y = np.load(ds_path+'Y_MFCC_63_128.npy')

x = x.reshape(x.shape[0], time, input_shape, 1)

x_train = x[:train_num]
y_train = y[:train_num]
x_val = x[train_num:train_num + val_num]
y_val = y[train_num:train_num + val_num]
x_test = x[train_num + val_num:]
y_test = y[train_num + val_num:]

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

reg = 0.0005


def conv_bn(x, filter, kernel_size, strides=(1, 1), padding='same'):
    x = Conv2D(filter, kernel_size, padding=padding,
               strides=strides, kernel_regularizer=l2(reg))(x)
    x = BatchNormalization()(x)
    return x


def conv_bn_relu(x, filter, kernel_size, strides=(1, 1), padding='same'):
    x = conv_bn(x, filter, kernel_size, strides=strides, padding=padding)
    x = Activation('relu')(x)
    return x


def conv_relu(x, filter, kernel_size, strides=(1, 1), padding='same'):
    x = Conv2D(filter, kernel_size, padding=padding,
               strides=strides, kernel_regularizer=l2(reg))(x)
    x = Activation('relu')(x)
    return x


def td_conv1d_relu(input, filter, kernel_size=15, strides=2, padding='same', conv_name=None, td_name=None, input_shape_conv=None, input_shape_td=None):
    x = TimeDistributed(Conv1D(filter, kernel_size=kernel_size, padding=padding, strides=strides,
                               kernel_regularizer=l2(reg), activation='relu',
                               name=conv_name),
                        name=td_name)(input)
    return x


print('done')

input = Input(shape=(time, input_shape, 1), name='input')  # 16,8
x = td_conv1d_relu(input, 64, conv_name='c1d1', td_name='c1d_td1')
x = td_conv1d_relu(x, 96, conv_name='c1d2', td_name='c1d_td2')
# x = TimeDistributed(MaxPooling1D(4,4), name = 'mp_td1')(x) #4, 2
x = td_conv1d_relu(x, 128, conv_name='c1d3', td_name='c1d_td3')
x = td_conv1d_relu(x, 160, conv_name='c1d4', td_name='c1d_td4')
x = td_conv1d_relu(x, 256, conv_name='c1d5', td_name='c1d_td5')
# x = TimeDistributed(MaxPooling1D(2,2), name = 'mp_td2')(x) #2,1
x = TimeDistributed(GlobalAveragePooling1D(name='gap1d'))(x)
x = TimeDistributed(Flatten(name='f'), name='td2')(x)
x = LSTM(256, dropout=0.5, recurrent_dropout=0.5,
         return_sequences=True, name='lstm_1')(x)
x = LSTM(128, dropout=0.5, recurrent_dropout=0.5,
         return_sequences=True, name='lstm_2')(x)
x = LSTM(64, dropout=0.5, recurrent_dropout=0.5, name='lstm_3')(x)

output = Dense(num_classes, activation='softmax', name='danse_out')(x)

model = Model(inputs=input, outputs=output)

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()
print('done')

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=50,
          verbose=1,
          validation_data=(x_val, y_val))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
model.save('.//nnmodels//cnn_no_mp_lstm_a'+str(int(100*score[1]))+'.h5')
model.save_weights('.//nnmodels//cnn_no_mp_lstm_a' +
                   str(int(100*score[1]))+'_weight.h5')
