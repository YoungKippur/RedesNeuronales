from pickle import FALSE
from matplotlib import units
import tensorflow as tf
import numpy as np

# Le enseñamos a resolver la ecuacion lineal f(x)=2x-4

# Input Values
input = np.array([1,2,3,4,5,6,7,8],dtype=int)

# Output Values 
output = np.array([-2,0,2,4,6,8,10,12], dtype=int)

if __name__ == "__main__":
    modelo = tf.keras.Sequential()
    modelo.add(tf.keras.layers.Dense(units=1, input_shape=[1]))

    modelo.compile(
        optimizer = tf.keras.optimizers.Adam(0.1),
        loss = 'mean_squared_error'
    )

    # epochs --> Cantidad de pruebas
    print('Entrenando... ;-)')
    historial = modelo.fit(input, output, epochs=500, verbose=True)
    print('Entrenado. (●*◡*●)')

    # Prerdiccion!!
    prediction = modelo.predict([-4])
    print('Resultado de la prediccion: ' + str(prediction))
