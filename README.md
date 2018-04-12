# Neural-Network-Model
Desarrollo del modelo general de red neural que planeo usar en algunos proyectos futuros.

Soy consciente que el modelo aquí propuesto se aleja mucho de ser bueno y que en internet hay modelos mil veces mejores. El objetivo que me planteo al programar y postear este código es aprender a manejarme en el machine learning. Dicho esto, se agradecen sugerencias y/o críticas :)
########################################################################################################################################
He escrito este código con los conocimientos que he obtenido siguiendo el curso online "Neural Networks and Deep Learning" de Andrew NG en coursera, por ello tiene similitudes con algunas funciones de los ejercicios de ese curso.

La base de datos utilizada para entrenar el modelo es la "MNIST DATABASE of handwritten digits".
########################################################################################################################################
Sobre NeuralModel_V1_0:

Cuando se utiliza la clase NeuralModel_V1_0 se le debe pasar como argumento una lista tipo ["124", "t23", "15", "s27"] donde el número de elementos de la lista es el número de capas, la letra indica la activación en cada capa y el número es el número de nodos en dicha capa.
El código es el siguiente:

#Número sin letra: Activación lineal. Recomendado para salidas en las que se busque un número real cualquiera.
#t: Tangente hiperbólica. Es la función que mejor resultado me ha dado en las capas intermedias con este modelo.
#r: Función ReLU. Me está dando problemas de overflow y división por cero.
#l: Leaky ReLU pendiente 0.01. Sin testear, imagino que pasará lo mismo que con la ReLU.
#s: Función sigmoide. Recomendado para salidas binarias.

Nota: El primer elemento de la lista es el número de inputs (no se considera capa) por lo que su función de activación es ignorada.
Nota2: Obsérvese que TODOS los elementos de la lista deben ser strings. Un elemento de otro tipo dará error.

Valores:
.layers: Lista de enteros con el número de nodos por capa.
.activations: Lista con la activación de cada capa.
.parameters: Diccionario con los parámetros "W" y "b".

Métodos:
.exe(X): Ejecuta la red para los valores de entrada X. Devuelve caché, AL, donde caché es una lista con la activación de cada capa y AL es la activación de la última capa únicamente (el output).
.train(X, Y, num_iterations, learning_rate): Calibra los parámetros num_iteraciones veces con ratio de aprendizaje learning_rate.
.test_bin(X, Y): Compara la salida binaria de la red con input X y lo compara con los resultados reales almacenados en Y. Hace un print con el porcentaje de acierto de la red.
########################################################################################################################################
Versiones futuras:

Para la próxima acualización trataré de arreglar el bug de la relu e implementaré los métodos .save() y .load() para poder guardar y cargar redes ya entrenadas.
