import time

import keras
from keras.layers import Conv2D, MaxPooling2D, Input
from keras.models import Model


class KerasModel(object):
    def __init__(self, name):
        self.name = name
        self.model = None

    def __enter__(self):
        self.start = time.clock()
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.took = (time.clock() - self.start) * 1000.0
        print('Model Name: {}, took {:.3f} ms to build'.format(self.name, self.took))
        self.model.summary()


def main():
    with KerasModel('Inception') as m:
        input_img = Input(shape=(256, 256, 3))

        tower_1 = Conv2D(64, (1, 1), padding='same', activation='relu')(input_img)
        tower_1 = Conv2D(64, (3, 3), padding='same', activation='relu')(tower_1)

        tower_2 = Conv2D(64, (1, 1), padding='same', activation='relu')(input_img)
        tower_2 = Conv2D(64, (5, 5), padding='same', activation='relu')(tower_2)

        tower_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(input_img)
        tower_3 = Conv2D(64, (1, 1), padding='same', activation='relu')(tower_3)

        output = keras.layers.concatenate([tower_1, tower_2, tower_3], axis=1)
        m.model = Model(inputs=input_img, outputs=output)

    with KerasModel('Residual') as m:
        # input tensor for a 3-channel 256x256 image
        x = Input(shape=(256, 256, 3))
        # 3x3 conv with 3 output channels (same as input channels)
        y = Conv2D(3, (3, 3), padding='same')(x)
        # this returns x + y.
        z = keras.layers.add([x, y])

        m.model = Model(inputs=x, outputs=z)


if __name__ == '__main__':
    main()
