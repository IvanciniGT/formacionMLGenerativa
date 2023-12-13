# XOR

Es un problema no lineal. No se puede resolver con una red neuronal de una sola capa.
Creamos una red con 2 capas. La primera capa tiene 2 neuronas y la segunda capa tiene 1 neurona.
Las neuronas de la primera capa utilizan la función de activación sigmoide y la neurona de la segunda capa también utiliza la función de activación sigmoide.
Las neuronas de la primera capa ayudan a aprender la función AND y la función OR. La neurona de la segunda capa aprende la función XOR utilizando las salidas de las neuronas de la primera capa como entradas.

## Red neuronal utilizada
```py
import numpy as np

# Datos de entrada y salida para la función XOR
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
outputs_xor = np.array([[0], [1], [1], [0]])  # Salida deseada para la función XOR

# Inicialización de pesos y sesgos
input_size = 2
hidden_size = 2
output_size = 1

weights_input_hidden = np.random.rand(input_size, hidden_size)
bias_hidden = np.random.rand(1, hidden_size)

weights_hidden_output = np.random.rand(hidden_size, output_size)
bias_output = np.random.rand(1, output_size)

# Tasa de aprendizaje
learning_rate = 0.1

# Función de activación sigmoide
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivada de la función sigmoide
def sigmoid_derivative(x):
    return x * (1 - x)

# Entrenamiento de la red
epochs = 10000
for epoch in range(epochs):
    # Propagación hacia adelante
    hidden_layer_input = np.dot(inputs, weights_input_hidden) + bias_hidden
    hidden_layer_output = sigmoid(hidden_layer_input)

    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
    predicted_output = sigmoid(output_layer_input)

    # Cálculo del error
    error = outputs_xor - predicted_output

    # Retropropagación
    output_delta = error * sigmoid_derivative(predicted_output)
    hidden_layer_error = output_delta.dot(weights_hidden_output.T)
    hidden_layer_delta = hidden_layer_error * sigmoid_derivative(hidden_layer_output)

    # Actualización de pesos y sesgos
    weights_hidden_output += hidden_layer_output.T.dot(output_delta) * learning_rate
    bias_output += np.sum(output_delta, axis=0, keepdims=True) * learning_rate

    weights_input_hidden += inputs.T.dot(hidden_layer_delta) * learning_rate
    bias_hidden += np.sum(hidden_layer_delta, axis=0, keepdims=True) * learning_rate

# Evaluación de la red entrenada
hidden_layer_output = sigmoid(np.dot(inputs, weights_input_hidden) + bias_hidden)
predicted_output = sigmoid(np.dot(hidden_layer_output, weights_hidden_output) + bias_output)

for i in range(len(inputs)):
    result = 1 if predicted_output[i] >= 0.5 else 0
    print(f"Input: {inputs[i]}, Predicted Output: {result}")

```

## Explicación del algoritmo con la red neuronal utilizada

**Paso 1: Propagación hacia adelante**

- Calcular la entrada de una neurona en la capa oculta:
  - `hidden_layer_input = (input1 * weight1) + (input2 * weight2) + bias`
  - Esto es simplemente una suma ponderada de las entradas (`input1` y `input2`) multiplicadas por los pesos (`weight1` y `weight2`) y luego sumando el sesgo (`bias`).

- Calcular la salida de una neurona en la capa oculta utilizando la función sigmoide:
  - `hidden_layer_output = sigmoid(hidden_layer_input)`
  - La función sigmoide se utiliza como función de activación para transformar la entrada en un valor entre 0 y 1. Esto introduce la no linealidad en la red.

- Calcular la entrada de la capa de salida:
  - `output_layer_input = (hidden_layer_output1 * weight_hidden_output1) + (hidden_layer_output2 * weight_hidden_output2) + bias_output`
  - Nuevamente, esto es una suma ponderada de las salidas de la capa oculta multiplicadas por los pesos de la capa de salida y luego sumando el sesgo de la capa de salida.

- Calcular la salida de la capa de salida utilizando la función sigmoide:
  - `predicted_output = sigmoid(output_layer_input)`
  - La función sigmoide se utiliza aquí para obtener una salida en el rango de 0 a 1.

**Paso 2: Cálculo del error**

- Calcular el error en la capa de salida:
  - `error = outputs_deseados - predicted_output`
  - El error es simplemente la diferencia entre la salida deseada (`outputs_deseados`) y la salida predicha (`predicted_output`). Este error mide cuánto se desvía la red de la salida deseada.

**Paso 3: Retropropagación**

- Calcular el delta de la capa de salida:
  - `output_delta = error * sigmoid_derivative(predicted_output)`
  - El delta de la capa de salida se calcula multiplicando el error por la derivada de la función sigmoide de la salida de la capa de salida. Esto cuantifica cuánto debe ajustarse la capa de salida para reducir el error.

- Calcular el error en la capa oculta:
  - `hidden_layer_error = output_delta.dot(weights_hidden_output.T)`
  - El error en la capa oculta se calcula propagando hacia atrás el delta de la capa de salida a través de los pesos de la capa de salida. Esto nos dice cómo contribuye cada neurona de la capa oculta al error de la capa de salida.

- Calcular el delta de la capa oculta:
  - `hidden_layer_delta = hidden_layer_error * sigmoid_derivative(hidden_layer_output)`
  - El delta de la capa oculta se calcula multiplicando el error en la capa oculta por la derivada de la función sigmoide de la salida de la capa oculta. Esto cuantifica cuánto debe ajustarse la capa oculta para reducir el error en la capa de salida.

**Paso 4: Actualización de pesos y sesgos**

- Actualizar los pesos de la capa de salida:
  - `weights_hidden_output += hidden_layer_output.T.dot(output_delta) * learning_rate`
  - Los pesos de la capa de salida se actualizan utilizando el producto de la transpuesta de la salida de la capa oculta (`hidden_layer_output.T`), el delta de la capa de salida (`output_delta`) y la tasa de aprendizaje (`learning_rate`). Esto ajusta los pesos para reducir el error en la capa de salida.

- Actualizar el sesgo de la capa de salida:
  - `bias_output += np.sum(output_delta, axis=0, keepdims=True) * learning_rate`
  - El sesgo de la capa de salida se actualiza utilizando la suma del delta de la capa de salida (`output_delta`) a lo largo del eje 0 (para todas las neuronas de la capa de salida) y la tasa de aprendizaje (`learning_rate`). Esto ajusta el sesgo para reducir el error en la capa de salida.

- Actualizar los pesos de la capa oculta:
  - `weights_input_hidden += inputs.T.dot(hidden_layer_delta) * learning_rate`
  - Los pesos de la capa oculta se actualizan utilizando el producto de la transpuesta de las entradas (`inputs.T`), el delta de la capa oculta (`hidden_layer_delta`) y la tasa de aprendizaje (`learning_rate`). Esto ajusta los pesos para reducir el error en la capa oculta.

- Actualizar el sesgo de la capa oculta:
  - `bias_hidden += np.sum(hidden_layer_delta, axis=0, keepdims=True) * learning_rate`
  - El sesgo de la capa oculta se actualiza utilizando la suma del delta de la capa oculta (`hidden_layer_delta`) a lo largo del eje 0 (para todas las neuronas de la capa oculta) y la tasa de aprendizaje (`learning_rate`). Esto ajusta el sesgo para reducir el error en la capa oculta.

Básicamente, en este paso, actualizamos los pesos y sesgos de ambas capas de la red utilizando el método del descenso de gradiente. Calculamos el gradiente de la función de pérdida con respecto a los parámetros de la red y ajustamos los parámetros en la dirección que minimiza la función de pérdida.

**Paso 5: Repetir los Pasos 1-4**

- El proceso de retropropagación se repite durante varias iteraciones (épocas) para ajustar gradualmente los pesos y sesgos de la red y reducir el error de predicción. En cada época, se realizan los Pasos 1-4 nuevamente con un nuevo lote de datos de entrenamiento hasta que la red converja a una solución adecuada.

## Justificación de las fórmulas usadas en el paso 4 - BACKPROPAGATION

Para actualizar los pesos y sesgos de la red neuronal, utilizamos el método del descenso de gradiente, que se basa en la derivada de la función de pérdida con respecto a los parámetros (pesos y sesgos) de la red. El objetivo es minimizar la función de pérdida ajustando los parámetros en la dirección en la que la pérdida disminuye más rápido.

**Actualización de pesos de la capa de salida:**
La actualización de los pesos de la capa de salida se realiza utilizando el gradiente descendiente. La derivada parcial de la función de pérdida con respecto a un peso específico se calcula utilizando la regla de la cadena:

$$\Delta w_{ij} = -\eta \frac{\partial L}{\partial w_{ij}}$$

Donde:
- $\Delta w_{ij}$ es el cambio en el peso $w_{ij}$.
- $\eta$ es la tasa de aprendizaje.
- $\frac{\partial L}{\partial w_{ij}}$ es la derivada parcial de la función de pérdida $L$ con respecto al peso $w_{ij}$.

La derivada parcial se calcula utilizando el error de la capa de salida y la derivada de la función de activación sigmoide:

$$\frac{\partial L}{\partial w_{ij}} = -\delta_j \cdot a_i$$

Donde:
- $\delta_j$ es el delta de la capa de salida para la neurona $j$.
- $a_i$ es la activación de la neurona de la capa oculta $i$.

Entonces, la actualización del peso $w_{ij}$ se realiza de la siguiente manera:

$$w_{ij_{\text{nuevo}}} = w_{ij_{\text{viejo}}} - \eta \cdot \delta_j \cdot a_i$$

Donde:
- $w_{ij_{\text{nuevo}}}$ es el nuevo valor del peso $w_{ij}$ después de la actualización.
- $w_{ij_{\text{viejo}}}$ es el valor actual del peso $w_{ij}$.

**Actualización de sesgos de la capa de salida:**
La actualización del sesgo de la capa de salida se realiza de manera similar. La derivada parcial de la función de pérdida con respecto al sesgo $b_j$ es simplemente el delta de la capa de salida $\delta_j$:

$$\Delta b_j = -\eta \cdot \delta_j$$

La actualización del sesgo $b_j$ se realiza de la siguiente manera:

$$b_{j_{\text{nuevo}}} = b_{j_{\text{viejo}}} - \eta \cdot \delta_j$$

**Actualización de pesos de la capa oculta:**
La actualización de los pesos de la capa oculta se realiza de manera similar a la capa de salida. La derivada parcial de la función de pérdida con respecto a un peso específico $v_{ij}$ se calcula utilizando el error de la capa oculta y la derivada de la función de activación sigmoide:

$$\frac{\partial L}{\partial v_{ij}} = -\delta_i \cdot x_j$$

Donde:
- $\delta_i$ es el delta de la capa oculta para la neurona $i$.
- $x_j$ es la entrada correspondiente a la neurona $j$ en la capa de entrada.

La actualización del peso $v_{ij}$ se realiza de la siguiente manera:

$$v_{ij_{\text{nuevo}}} = v_{ij_{\text{viejo}}} - \eta \cdot \delta_i \cdot x_j$$

Donde:
- $v_{ij_{\text{nuevo}}}$ es el nuevo valor del peso $v_{ij}$ después de la actualización.
- $v_{ij_{\text{viejo}}}$ es el valor actual del peso $v_{ij}$.

**Actualización de sesgos de la capa oculta:**
La actualización del sesgo de la capa oculta se realiza de manera similar a la capa de salida. La derivada parcial de la función de pérdida con respecto al sesgo $c_i$ es simplemente el delta de la capa oculta $\delta_i$:

$$\Delta c_i = -\eta \cdot \delta_i$$

La actualización del sesgo $c_i$ se realiza de la siguiente manera:

$$c_{i_{\text{nuevo}}} = c_{i_{\text{viejo}}} - \eta \cdot \delta_i$$

Estas fórmulas se derivan utilizando el método del descenso de gradiente y la regla de la cadena para calcular las derivadas parciales de la función de pérdida con respecto a los parámetros de la red. Luego, se ajustan los pesos y sesgos en la dirección que minimiza la función de pérdida.

## Ejemplo de cálculo de la primera iteración:

**Valores iniciales aleatorios que nos imaginamos que el programa ha generado:**

- Pesos de la capa oculta (weights_input_hidden): 
  - weight1 = 0.5
  - weight2 = 0.5

- Sesgo de la capa oculta (bias_hidden): 
  - bias_hidden = 1

- Pesos de la capa de salida (weights_hidden_output): 
  - weight_hidden_output1 = 0.5
  - weight_hidden_output2 = 0.5

- Sesgo de la capa de salida (bias_output): 
  - bias_output = 1

- Tasa de aprendizaje (learning_rate): 
  - learning_rate = 0.1

**Entradas de la red (inputs):**
- Para la primera iteración:
  - input1 = 0
  - input2 = 0
  - output_esperado = 0

**Paso 1: Propagación hacia adelante**

- Calcular la entrada de una neurona en la capa oculta:
  - `hidden_layer_input = (input1 * weight1) + (input2 * weight2) + bias_hidden`
  - `hidden_layer_input = (0 * 0.5) + (0 * 0.5) + 1`
  - `hidden_layer_input = 1`

- Calcular la salida de una neurona en la capa oculta utilizando la función sigmoide:
  - `hidden_layer_output = sigmoid(hidden_layer_input)`
  - `hidden_layer_output = sigmoid(1)`
  - `hidden_layer_output ≈ 0.731`

- Calcular la entrada de la capa de salida:
  - `output_layer_input = (hidden_layer_output1 * weight_hidden_output1) + (hidden_layer_output2 * weight_hidden_output2) + bias_output`
  - `output_layer_input = (0.731 * 0.5) + (0.731 * 0.5) + 1`
  - `output_layer_input ≈ 1.366`

- Calcular la salida de la capa de salida utilizando la función sigmoide:
  - `predicted_output = sigmoid(output_layer_input)`
  - `predicted_output = sigmoid(1.366)`
  - `predicted_output ≈ 0.796`

**Paso 2: Cálculo del error**

- Calcular el error en la capa de salida:
  - `error = output_deseado - predicted_output`
  - `error = 0 - 0.796`
  - `error ≈ -0.796`

**Paso 3: Retropropagación**

- Calcular el delta de la capa de salida:
  - `output_delta = error * sigmoid_derivative(predicted_output)`
  - `output_delta = -0.796 * sigmoid_derivative(0.796)`
  - `output_delta ≈ -0.175`

- Calcular el error en la capa oculta:
  - `hidden_layer_error = output_delta.dot(weights_hidden_output.T)`
  - `hidden_layer_error ≈ -0.175 * (0.5, 0.5)`
  - `hidden_layer_error ≈ (-0.0875, -0.0875)`

- Calcular el delta de la capa oculta:
  - `hidden_layer_delta = hidden_layer_error * sigmoid_derivative(hidden_layer_output)`
  - `hidden_layer_delta ≈ (-0.0875, -0.0875) * sigmoid_derivative(0.731)`
  - `hidden_layer_delta ≈ (-0.0875 * 0.196, -0.0875 * 0.196)`
  - `hidden_layer_delta ≈ (-0.017, -0.017)`

**Paso 4: Actualización de pesos y sesgos**

- Actualizar los pesos de la capa de salida:
  - `weights_hidden_output += hidden_layer_output.T.dot(output_delta) * learning_rate`
  - `weights_hidden_output += (0.731, 0.731) * (-0.175) * 0.1`
  - `weights_hidden_output ≈ (0.731 - 0.0126, 0.731 - 0.0126)`
  - `weights_hidden_output ≈ (0.7184, 0.7184)`

- Actualizar el sesgo de la capa de salida:
  - `bias_output += np.sum(output_delta, axis=0, keepdims=True) * learning_rate`
  - `bias_output += -0.175 * 0.1`
  - `bias_output ≈ 0.9825`

- Actualizar los pesos de la capa oculta:
  - `weights_input_hidden += inputs.T.dot(hidden_layer_delta) * learning_rate`
  - `weights_input_hidden += (0, 0).T.dot((-0.017, -0.017)) * 0.1`
  - `weights_input_hidden += (0, 0) * (0, 0)`
  - `weights_input_hidden ≈ (0.5, 0.5)`

- Actualizar el sesgo de la capa oculta:
  - `bias_hidden += np.sum(hidden_layer_delta, axis=0, keepdims=True) * learning_rate`
  - `bias_hidden += (-0.017, -0.017) * 0.1`
  - `bias_hidden ≈ (0.9983, 0.9983)`

En esta iteración, hemos calculado y actualizado los deltas de ambas capas de la red, así como los pesos y sesgos. Estos valores actualizados se utilizarán en la próxima iteración para continuar ajustando la red.






