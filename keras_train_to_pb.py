# dataset: https://github.com/zalandoresearch/fashion-mnist
# https://medium.com/@pipidog/how-to-convert-your-keras-models-to-tensorflow-e471400b886a

import keras
import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D, Input, Dense, Dropout, Flatten
from keras.models import Model

from utils import utils
from utils.mnist_reader import load_mnist

LESS_DATA = True
BATCH_SIZE = 100
EPOCHS = 2


def main():
    x_train, y_train = load_mnist('data/fashion', kind='train')
    x_test, y_test = load_mnist('data/fashion', kind='t10k')

    if LESS_DATA:
        x_train, y_train = x_train[0:10000, ], y_train[0:10000, ]
        x_test, y_test = x_train[0:1000, ], y_train[0:1000, ]

    num_classes = len(set(y_train))

    # One-hot encoding
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    input_img = Input(shape=(28, 28, 1))
    x = Conv2D(32, kernel_size=(3, 3), activation='relu')(input_img)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=input_img, outputs=output)
    model.summary()

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=BATCH_SIZE,
              epochs=EPOCHS,
              verbose=1,
              validation_data=(x_test, y_test))

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # convert keras model to pb files
    output_names = [out.op.name for out in model.outputs]
    frozen_graph = utils.freeze_session(keras.backend.get_session(),
                                        output_names=output_names)
    tf.train.write_graph(frozen_graph, "output_models/", "keras_fashion.pb", as_text=False)
    tf.train.write_graph(frozen_graph, "output_models/", "keras_fashion.pbtxt", as_text=True)


if __name__ == '__main__':
    main()
