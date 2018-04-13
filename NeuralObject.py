import numpy as np


def sigmoid(z):
    s = 1./(1+np.exp(-z))
    return s

def relu(z):
    return z * (z > 0)

def leaky(z):
    return z * (z > 0) + z * 0.01 * (z <= 0)


def initialize_parameters(layers):
    parameters = {}
    L = len(layers)
    for l in range(1, L):
        parameters["W" + str(l)] = np.random.randn(layers[l], layers[l - 1]) * 0.01
        parameters["b" + str(l)] = np.zeros((layers[l], 1))
    return parameters

def linear_forward(a, W, b):
    z = np.dot(W, a) + b
    return z

def apply_activation(z, activation):
    if activation == "lin":
        return z
    elif activation == "sig":
        a = sigmoid(z)
    elif activation == "relu":
        a = relu(z)
    elif activation == "leaky":
        a = leaky(z)
    elif activation == "tanh":
        a = np.tanh(z)
    return a

def derived_activation(a, activation):
    zeros = np.zeros(a.shape)

    if activation == "lin":
        return zeros + 1.
    elif activation == "sig":
        d = a*(1. - a)
    elif activation == "relu":
        d = (a > 0)*1.
    elif activation == "leaky":
        d = (a <= 0)*0.01 + (a > 0)*1.
    elif activation == "tanh":
        d = 1. - a * a
    return d

def backprop(dAl, activations, cache, parameters, l):
    grads = {}
    m = dAl.shape[1]
    g_der = derived_activation(cache["A" + str(l)], activations[l])

    dZ = np.multiply(dAl, g_der)
    dW = 1/m * np.dot(dZ, cache["A" + str(l-1)].T)
    db = 1/m * np.sum(dZ, axis = 1, keepdims = True)
    dAl_prev = np.dot(parameters["W" + str(l)].T, dZ)

    grads["dA" + str(l-1)] = dAl_prev
    grads["dW"+ str(l)] = dW
    grads["db" + str(l)] = db
    return grads

def coste(AL, Y):
    m = Y.shape[1]
    coste = -1 / m * np.sum(Y * np.log(AL) + (1 - Y) * np.log(1.00001 - AL))
    return coste

def write_array(array, file):
    for i in range(len(array)):
        file.write(str(array[i]) + " ")
    file.write("\n")

def read_array(file, f = False):
    line = file.readline()
    if f:
        return list(map(float, line.split()))
    else:
        return list(map(int, line.split()))


class NeuralModel_V1_0:

    def __init__(self, raw_layers):
        layers = []
        activations = []
        num = "123456789"

        for i in range(len(raw_layers)):
            if raw_layers[i][0] in num:
                activations.append("lin")
                layers.append(int(raw_layers[i]))
            else:
                layers.append(int(raw_layers[i][1:]))
                if raw_layers[i][0] == "s":
                    activations.append("sig")
                elif raw_layers[i][0] == "r":
                    activations.append("relu")
                elif raw_layers[i][0] == "t":
                    activations.append("tanh")
                elif raw_layers[i][0] == "l":
                    activations.append("leaky")
                else:
                    raise ValueError("Non valid list of layers.")

        self.parameters = initialize_parameters(layers)
        self.activations = activations
        self.layers = layers

    def exe(self, X):
        parameters = self.parameters
        activations = self.activations
        cache = {}

        L = len(parameters) // 2
        cache["A0"] = X
        for l in range(1, L + 1):
            z = linear_forward(cache["A" + str(l-1)], parameters["W" + str(l)], parameters["b" + str(l)])
            cache["A" + str(l)] = apply_activation(z, activations[l])

        return cache, cache["A" + str(L)]

    def train(self, X, Y, num_iterations, learning_rate):
        parameters = self.parameters
        activations = self.activations
        L = len(parameters) // 2

        for i in range(num_iterations):

            cache, AL = self.exe(X)

            dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1.00001 - AL))

            for l in reversed(range(1, L + 1)):
                grads = backprop(dAL, activations, cache, parameters, l)

                parameters["W" + str(l)] -= learning_rate * grads["dW" + str(l)]
                parameters["b" + str(l)] -= learning_rate * grads["db" + str(l)]

                dAL = grads["dA" + str(l-1)]

            if ((i + 1) % 100) == 0:
                print("El coste tras %d iteraciones es %f" % (i + 1, coste(AL, Y)))

    def test_bin(self, X, Y):
        count = 0.0
        m = Y.shape[1]
        cache, raw_ans = self.exe(X)
        ans = (raw_ans > 0.5)*1

        for i in range(m):
            if all(ans[:, i] == Y[:, i]):
                count += 1

        accuracy = count*100/m

        print(("El porcentaje de acierto es del %.2f" % accuracy) + '%')

    def save(self, file_name = "NN_save"):
        L = len(self.layers)

        with open(file_name, "w") as file:
            write_array(self.layers, file)
            write_array(self.activations, file)

            for l in range(1, L):
                write_array(self.parameters["b" + str(l)].T[0], file)

            for l in range(1, L):
                for i in range(len(self.parameters["W" + str(l)])):
                    write_array(self.parameters["W" + str(l)][i], file)

    def load(self, file_name = "NN_save"):
        with open(file_name, "r") as file:
            self.layers = read_array(file)
            self.activations = file.readline().split()
            L = len(self.layers)

            for l in range(1, L):
                self.parameters["b" + str(l)] = np.array(read_array(file, True), ndmin = 2).T

            for l in range(1, L):
                par = []
                for n in range(self.layers[l]):
                    par.append(read_array(file, True))
                self.parameters["W" + str(l)] = np.array(par)
