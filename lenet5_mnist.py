"""
Treina uma CNN com o dataset MNIST.

A CNN é inspirada na arquitetura LeNet-5, com algumas
alterações nas funções de ativação, padding e pooling.
"""

# importar pacotes necessários
from keras.utils import to_categorical
from keras.optimizers import SGD
from keras import backend
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
from cnn import LeNet5     # ESTA É A CLASSE CRIADA POR NÓS

# importar e normalizar o dataset MNIST
dataset = fetch_mldata("MNIST Original")
labels = dataset.target
data = dataset.data.astype("float32") / 255.0

# converter as imagens de 1D para o formato (28x28x1)
if backend.image_data_format() == "channels_last":
    data = data.reshape((data.shape[0], 28, 28, 1))
else:
    data = data.reshape((data.shape[0], 1 , 28, 28))

# dividir o dataset entre train (75%) e test (25%)
(trainX, testX, trainY, testY) = train_test_split(data, labels)

# Transformar labels em vetores binarios
trainY = to_categorical(trainY, 10)
testY = to_categorical(testY, 10)

# inicializar e otimizar modelo
print("[INFO] inicializando e otimizando a CNN...")
model = LeNet5.build(28, 28, 1, 10)
model.compile(optimizer=SGD(0.01), loss="categorical_crossentropy",
              metrics=["accuracy"])

# treinar a CNN
print("[INFO] treinando a CNN...")
H = model.fit(trainX, trainY, batch_size=128, epochs=20, verbose=2,
          validation_data=(testX, testY))

# avaliar a CNN
print("[INFO] avaliando a CNN...")
predictions = model.predict(testX, batch_size=64)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1),
                            target_names=[str(label) for label in range(10)]))

# plotar loss e accuracy para os datasets 'train' e 'test'
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0,20), H.history["loss"], label="train_loss")
plt.plot(np.arange(0,20), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0,20), H.history["acc"], label="train_acc")
plt.plot(np.arange(0,20), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig('cnn.png', bbox_inches='tight')









