from NeuralObject import *
import random
from mnistload import *

batches = []
np.random.seed(1)

#CARGA LAS IMAGENES DEL MNIST EN MATRICES
print('\nCargando los sets...')
X_test, Y_test = load_set("test")
X_train, Y_train = load_set("train")

#SEPARA EN PAQUETES
for i in range(1, 31): #Separa las 60000 imágenes en paquetes de 2000
    batches.append((X_train[:, ((i-1)*2000):(i*2000)], Y_train[:, ((i-1)*2000):(i*2000)]))
random.shuffle(batches)

#GENERA LOS PARÁMETROS DE LA RED EN FUNCION DE LA LISTA DE CAPAS
print("Sets cargados. Cargando red neural...")
Net = NeuralModel_V1_0(["784", "t50", "t30", "t16", "s10"]) #Red de tres capas ocultas con activación tanh y una capa de salida sigmoide.
print("Red cargada.")

#ENTRENAMIENTO DE LA RED
for b in range(30): #Número de paquetes utilizados en el entrenamiento
    print('\nEntrenando red en el batch %d' % (b+1))
    Net.train(batches[b][0], batches[b][1], num_iterations = 500, learning_rate = 0.7)

#TESTEO EN LAS IMAGENES DEL TEST
print('\nRed entrenada. Test:')
Net.test_bin(X_test, Y_test)
Net.save()
