'''
This neural network consumes imput features in shape: 1, 39368
'''
import keras
from keras.models import load_model
from keras.datasets import mnist
from keras.layers import Input, Dense, BatchNormalization, Activation, Dropout, Conv1D, MaxPooling1D, Flatten, GlobalAveragePooling1D
from keras.models import Model, Sequential
from keras.optimizers import Adam, Adadelta
from keras.regularizers import l2
import numpy as np
from keras.utils import plot_model


batch_size = 5  # 512
num_classes = 2
epochs = 200  # need more
train_num = 55
val_num = 5
preprocDataset = './preprocDataset/'
l2_param = 0.005

img_rows, img_cols = 1, 39368
input_shape = (39368, 1)
x = np.load(preprocDataset+'x_dataset_(65,39368).npy')
y = np.load(preprocDataset+'y_labels_(65,39368).npy')
x = x.reshape(x.shape[0],   img_cols, img_rows)
y = y.reshape(y.shape[0],  num_classes)

x_train = x[:train_num]
y_train = y[:train_num]
x_val = x[train_num:train_num + val_num]
y_val = y[train_num:train_num + val_num]
x_test = x[train_num + val_num:]
y_test = y[train_num + val_num:]

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')


model = Sequential()

model.add(Conv1D(32, kernel_size=15,
                 padding='same', strides=4,
                 input_shape=input_shape, name='conv_1d_1'))

model.add(Activation('relu'))
# model.add(MaxPooling1D(pool_size=2,strides=2,padding='same'))
model.add(Conv1D(64, 15,
                 padding='same', strides=4,
                 kernel_regularizer=l2(l2_param), name='conv_1d_2'))
model.add(Activation('relu'))

#model.add(Conv1D(64, 7, padding='same'))
# model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=5, strides=5, padding='same'))
model.add(Conv1D(128, 15, padding='same', strides=4,
                 kernel_regularizer=l2(l2_param),
                 name='conv_1d_3'))
model.add(Activation('relu'))

model.add(Conv1D(256, 15, padding='same', strides=4,
                 kernel_regularizer=l2(l2_param),
                 name='conv_1d_4'))
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=4, strides=4, padding='same'))


# model.add(Dropout(0.5))

# model.add(Dense(512))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))

# model.add(Dense(128))
model.add(GlobalAveragePooling1D())
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(num_classes, activation='softmax', name='dense_1'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_val, y_val))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
model.save('cnn_fds_50_l2_bs512')

model.summary()
plot_model(model, to_file='model.png')
