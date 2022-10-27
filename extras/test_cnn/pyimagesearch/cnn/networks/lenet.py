
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K

class LeNet:
    @staticmethod
    def build(numChannels, imgRows, imgCols, numClasses, activation="relu", weightsPath=None):
        #init the model
        model = Sequential()
        inputShape = (imgRows, imgCols, numChannels)

        if K.image_data_format() == "channels_first":
            inputShape = (numChannels, imgRows, imgCols)
        
        #define the first set of CONV => ACTIVATION => POOL layers
        model.add(Conv2D(20, 5, padding="same", input_shape=inputShape))
        model.add(Activation(activation))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        #define the second set of CONV => ACTIVATION => POOL layers
        model.add(Conv2D(50, 5, padding="same"))
        model.add(Activation(activation))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        #define the first FC => ACTIVATION layers
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation(activation))

        #define the second FC layer
        model.add(Dense(numClasses))

        #define the soft-max classifier
        model.add(Activation("softmax"))

        if weightsPath is not None:
            model.load_weights(weightsPath)
        
        return model

        