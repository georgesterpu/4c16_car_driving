import sys
sys.path.append('/usr/lib/python3.6/site-packages/')

import keras
from keras.layers import Input, Dense, Conv2D, Lambda, Flatten
from keras.regularizers import l2


def build_model():

    inputs = Input(shape=(66, 200, 3))

    x = inputs

    # Do the scaling outside the network, so as to maintain compatibility
    # of the compiled model across minor version differences of Python
    # (Lambda layers go to python bytecode, so they break that).
    # x = Lambda(lambda pel: pel/255*2-1)(x)

    x = Conv2D(24, [5, 5], strides=[2, 2], kernel_regularizer=l2(0.001), activation='elu')(x)
    x = Conv2D(36, [5, 5], strides=[2, 2], kernel_regularizer=l2(0.001), activation='elu')(x)
    x = Conv2D(48, [5, 5], strides=[2, 2], kernel_regularizer=l2(0.001), activation='elu')(x)

    x = Conv2D(64, [3, 3], kernel_regularizer=l2(0.001), activation='elu')(x)
    x = Conv2D(64, [3, 3], kernel_regularizer=l2(0.001), activation='elu')(x)

    x = Flatten()(x)
    x = Dense(100, kernel_regularizer=l2(0.001), activation='elu')(x)
    x = Dense(50, kernel_regularizer=l2(0.001), activation='elu')(x)
    x = Dense(10, kernel_regularizer=l2(0.001), activation='elu')(x)

    output = Dense(1)(x)

    model = keras.models.Model(inputs=inputs, outputs=output)
    model.summary()
    model.compile(optimizer='adam', loss='mse')

    return model
