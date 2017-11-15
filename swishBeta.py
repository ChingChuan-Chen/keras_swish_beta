#!/usr/bin/env python
import keras
from keras import backend as K
from keras.datasets import mnist
from keras.layers import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling1D
from keras.layers import BatchNormalization
from keras.layers import initializers, InputSpec
from keras.models import Sequential
from keras.utils import multi_gpu_model
from keras.engine.topology import Layer

class SwishBeta(Layer):
    def __init__(self, trainableBeta = False, beta_initializer = 'ones', **kwargs):
        super(SwishBeta, self).__init__(**kwargs)
        self.supports_masking = True
        self.beta_initializer = initializers.get(beta_initializer)
        
    def build(self, input_shape):
        self.beta = self.add_weight(shape=[1], name='beta', 
                                    initializer=self.beta_initializer)
        self.input_spec = InputSpec(ndim=len(input_shape))
        self.built = True

    def call(self, inputs):
        return inputs * K.sigmoid(self.beta * inputs)

    def get_config(self):
        config = {'beta_initializer': initializers.serialize(self.beta_initializer)}
        base_config = super(SwishBeta, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

num_classes = 10
img_rows, img_cols = 28, 28
img_rows_new, img_cols_new = 299, 299

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)
    
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255.
x_test /= 255.

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(64, kernel_size=(3, 3), padding = 'same', 
                 kernel_initializer = 'he_uniform', input_shape=input_shape))
model.add(BatchNormalization())
model.add(SwishBeta())
model.add(Conv2D(128, (3, 3), padding = 'same', 
                 kernel_initializer = 'he_uniform'))
model.add(BatchNormalization())
model.add(SwishBeta())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(256, (3, 3), padding = 'same', 
                 kernel_initializer = 'he_uniform'))
model.add(BatchNormalization())
model.add(SwishBeta())
model.add(Conv2D(256, (3, 3), padding = 'same', 
                 kernel_initializer = 'he_uniform'))
model.add(BatchNormalization())
model.add(SwishBeta())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(512, (3, 3), padding = 'same', 
                 kernel_initializer = 'he_uniform'))
model.add(BatchNormalization())
model.add(SwishBeta())
model.add(Conv2D(512, (3, 3), padding = 'same', 
                 kernel_initializer = 'he_uniform'))
model.add(BatchNormalization())
model.add(SwishBeta())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(GlobalAveragePooling2D())
model.add(SwishBeta())
model.add(Dense(num_classes, activation='softmax'))

# single gpu
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size = 128,
                    epochs = 500,
                    verbose = 1,
                    callbacks = [keras.callbacks.EarlyStopping(patience=4)],
                    validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
