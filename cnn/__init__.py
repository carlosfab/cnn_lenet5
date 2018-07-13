"""
Contém as implementações de arquiteturas de CNN.

[LeNet5] - CNN inspirada na arquitetura de LeCun [1], com algumas
alterações nas funções de ativação, padding e pooling.

[1] http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf
"""

# importar os pacotes necessários
from keras.models import Sequential
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation

class LeNet5(object):
    """
    Arquitetura LeNet5 com pequenas alterações.

    Com foco no reconhecimento de dígitos, esta CNN é composta
    por uma sequência contendo os seguintes layers:

    INPUT => CONV => POOL => CONV => POOL => FC => FC => OUTPUT
    """
    @staticmethod
    def build(width, height, channels, classes):
        """
        Constroi uma CNN com arquitetura LeNet5.

        :param width: Largura em pixel da imagem.
        :param height: Altura em pixel da imagem.
        :param channels: Quantidade de canais da imagem.
        :param classes: Quantidade de classes para o output.
        :return: Cnn do tipo LeNet5.
        """
        inputShape = (height, width, channels)

        model = Sequential()
        model.add(Conv2D(6, (5, 5), padding="same",
                         input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D((2,2)))
        model.add(Conv2D(16, (5, 5)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D((2,2)))
        model.add(Flatten())
        model.add(Dense(120))
        model.add(Activation("relu"))
        model.add(Dense(84))
        model.add(Activation("relu"))
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        return model
