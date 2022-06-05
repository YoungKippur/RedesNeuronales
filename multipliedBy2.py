from pickle import FALSE
from matplotlib import units
import tensorflow as tf
import numpy as np

# Le enseñaremos a multiplicar *2

# Input Values
input = np.array([2,4,5,6,7,8,9,10],dtype=int)

# Output Values 
output = np.array([4,8,10,12,14,16,18,20], dtype=int)

# Empezamos con la red neuronal (●'◡'●)
if __name__ == "__main__":
    modelo = tf.keras.Sequential()
    modelo.add(tf.keras.layers.Dense(units=1, input_shape=[1]))

    modelo.compile(
        optimizer = tf.keras.optimizers.Adam(0.1),
        loss = 'mean_squared_error'
    )

    # epochs --> Cantidad de pruebas
    print('Entrenando... ;-)')
    historial = modelo.fit(input, output, epochs=1000, verbose=False)
    print('Entrenado. (●*◡*●)')

    # Prerdiccion!!
    prediction = modelo.predict([14])
    print('Resultado de la prediccion: ' + str(prediction))



