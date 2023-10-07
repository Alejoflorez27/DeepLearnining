#load data
import numpy as np
import matplotlib.pyplot       as plt
from mlxtend.data import loadlocal_mnist
from sklearn.metrics import confusion_matrix
from tensorflow.keras.optimizers import SGD

X_train, y_train = loadlocal_mnist(
        images_path='MNIST/train-images-idx3-ubyte', 
        labels_path='MNIST/train-labels-idx1-ubyte')

X_test, y_test = loadlocal_mnist(
        images_path='MNIST/t10k-images-idx3-ubyte', 
        labels_path='MNIST/t10k-labels-idx1-ubyte')

labels,count_class = np.unique(y_train,return_counts=True)

print('[INFO] \nlabels: \n %s \ncount per class \n %s' % (labels,count_class))
print('Training set dimensions: %s x %s' % (X_train.shape[0],X_train.shape[1]))
print('Test set dimensions: %s x %s' % (X_test.shape[0], X_test.shape[1]))

#ver imagenes del dataset
dim1=28
dim2=28     
image = X_train[2000].reshape((28,28)) #redimensiona la imagen
print(y_train[2000]) #imprime el 5 que esta en la posicion 2000
plt.imshow(image,cmap="gray")

##importamos librerias para la construcción del modelo
#multilayer feedforward
from keras.models import Sequential
from keras.layers import Dense


##construccion del modelo
num_classes = labels.size ##tamaño de la clases
nu_hl1 = 400    #neuronas en la capa 1
nu_hl2 = 200    #neuronas en la capa 2
nu_hl3 = 100    #neuronas en la capa 3
nu_hl4 = num_classes    #total de clases MNIST


model = Sequential()
model.add(Dense(nu_hl1, input_dim=dim1*dim2, activation='relu'))
model.add(Dense(nu_hl2, activation='relu'))
model.add(Dense(nu_hl3, activation='relu'))
model.add(Dense(nu_hl4, activation='softmax'))  #sigmoid
model.summary()         #estructura del modelo

#lo que hace es coger una categoria volver las otra una sola para clasificar una a una
from keras.utils import to_categorical
train_labels = to_categorical(y_train, num_classes=num_classes)
test_labels = to_categorical(y_test, num_classes=num_classes)

print('convertir: ',y_train[0],' a one hot encoding : ',train_labels[0])


#'categorical_crossentropy' debido a que es una clasificacion multiclase
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, train_labels, epochs=5, batch_size=10)

#prediccion del modelo y de un dato de prueba
predictions = model.predict(X_test)
predictions[0]


print("etiqueta 10 primeras imagenes de prueba:  ",y_test[:10],\
      "\nprediccion 10 primeras imagenes de prueba:",predictions[:10])

print(y_test[1])

_, accuracy = model.evaluate(X_test, test_labels)
print('Accuracy: %.2f' % (accuracy*100))

'''
## Matriz de confusión
Out_CM = np.zeros([60000,1],dtype=np.float64)
#Para la matriz de confusión se necesita la posición de la neurona que tuvo mayor #activación, esto determina la clase
for i in range(0,60000):
    Out_CM[i] = np.argmax(predictions[i,:])

#Nombres de las clases para la matriz de confusión
class_names=[0,1,2,3,4,5,6,7,8,9]
'''

y_test=np.argmax(predictions, axis = 1)
test_labels=np.argmax(test_labels, axis = 1)

# Calculo de la matriz de confusión
cnf_matrix = confusion_matrix(test_labels, y_test)
print(cnf_matrix)