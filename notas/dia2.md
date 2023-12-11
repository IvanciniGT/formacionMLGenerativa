
# Redes neuronales

Dada una entrada de datos, conseguir una red neuronal (programa), que genere una salida.
 (x,y) aleatorios -> (X, Y) de forma que Y ~= sin(x)

```py
def red_neuronal(x,y):
    return (x,sin(x))
```

## Perceptrón

Es una función matemática que recibe un conjunto de datos de entrada (x1, x2, ... xn)
Combina esos datos para formar una nueva variable: X = w1*x1 + w2*x2 + ... + wn*xn + b

Y una vez calculada esa variable X, la pasa por una función de activación (sigmoide, tanh, relu, etc) para obtener la salida.

La salida depende de la función de activación que se utilice.

## Redes neuronales

Una red neuronal es un conjunto de perceptrones conectados entre sí.

### Arquitetura de una red neuronal

- Capa de entrada: Tenemos un conjunto de datos de entrada (x1, x2, ... xn)
- Capas ocultas (n): Tenemos un conjunto de perceptrones que reciben los datos de entrada y generan una salida (y1, y2, ... yn)
- Capa de salida: Tenemos un conjunto de datos de salida (y1, y2, ... yn)


    X1      P1      P4      P6

    X2      P2      P5      P7

    X3      P3              P8

    XN

P1 (Recibe X1, X2, X3, ..., XN) -> PY1
P2 (Recibe X1, X2, X3, ..., XN) -> PY2
P3 (Recibe X1, X2, X3, ..., XN) -> PY3

P4 (Recibe PY1, PY2, PY3) -> PY4
P5 (Recibe PY1, PY2, PY3) -> PY5

P6 (Recibe PY4, PY5) -> PY6
P7 (Recibe PY4, PY5) -> PY7
P8 (Recibe PY4, PY5) -> PY8

## Red neuronal que aprenda la función "Y LOGICO"

| X1 | X2 | Y | X | P1 | Error | x Learning Rate |
|----|----|---|---|--- | ----- | --- |
| 0  | 0  | 0 | 0.25 | 1 | -1 | -0.1 |
| 0  | 1  | 0 | 0.75 | 1 | -1 | -0.1 |
| 1  | 0  | 0 | 0.75 | 1 | -1 | -0.1 |
| 1  | 1  | 1 | 1.25 | 1 | 0 | 0 |

### Red que vamos a definir

    X1 --->P1 --> Y1
    X2 /

P1 (Recibe X1, X2) -> P1
    
        0.5     0.5     0.25
    X = w1*x1 + w2*x2 + b

    dot ( (x1, x2) , (w1, w2) ) = w1*x1 + w2*x2

    P1 = X > 0 ? 1 : 0

Qué valores pongo de w1, w2 y b para que los resultados que obtenga (P1) sean los propios de una función Y LOGICO 

El proceso de aprendizaje consiste en encontrar los valores de w1, w2 y b que mejor se ajusten a los datos de entrada, para producir la salida deseada.

Con esos mismos pesos (w1=0.5, w2=0.5) y con un bias -0.8

| X1 | X2 | Y | X | P1 | Error |
|----|----|---|---|--- | ----- |
| 0  | 0  | 0 | -0.8 | 0 | 0 |
| 0  | 1  | 0 | -0.05 | 0 | 0 |
| 1  | 0  | 0 | -0.05 | 0 | 0 |
| 1  | 1  | 1 | 0.2 | 1 | 0 |

# OR EXCLUSIVO (XOR)

| X1 | X2 | Y | 
|----|----|---|
| 0  | 0  | 0 |
| 0  | 1  | 1 |
| 1  | 0  | 1 |
| 1  | 1  | 0 |




    X1      P1(AND)     P3(XOR)
    X2      P2(OR)

    PX1 = p1w1*x1 + p1w2*x2 + p1b
    PY1 = F_Activación(PX1)
    PX2 = p2w1*x1 + p2w2*x2 + p2b
    PY2 = F_Activación(PX2)
    PX3 = p3w1*PY1 + p3w2*PY2 + p3b

    PX3 = p3w1*(F_Activación(p1w1*x1 + p1w2*x2 + p1b)) 
        + p3w2*(F_Activación(p2w1*x1 + p2w2*x2 + p2b))
        + p3b

    Datos a calcular tengo?
        p3w1
        p1w1
        p1w2
        p1b
        p3w2
        p2w1
        p2w2
        p2b
        p3b

## Propagación del recálculo de los pesos:

1. **Capa de Entrada (2 nodos):**
   - \(x_1\), \(x_2\) son las entradas.

2. **Capa Oculta (2 nodos):**
   - \(z_1 = w_{11} \cdot x_1 + w_{21} \cdot x_2 + b_1\)
   - \(z_2 = w_{12} \cdot x_1 + w_{22} \cdot x_2 + b_2\)
   - \(a_1 = \sigma(z_1)\) (donde \(\sigma\) es la función de activación, por ejemplo, la sigmoide)
   - \(a_2 = \sigma(z_2)\)

3. **Capa de Salida (1 nodo):**
   - \(z_{\text{salida}} = w_{\text{salida}} \cdot a_1 + w_{\text{salida}} \cdot a_2 + b_{\text{salida}}\)
   - \(a_{\text{salida}} = \sigma(z_{\text{salida}})\)

4. **Función de Pérdida (Binary Cross Entropy Loss):**
   - \(\text{Loss} = -[y \cdot \log(a_{\text{salida}}) + (1 - y) \cdot \log(1 - a_{\text{salida}})]\)

En el proceso de backpropagation, calculamos las derivadas parciales de la pérdida con respecto a los parámetros y luego actualizamos los parámetros utilizando el descenso de gradiente. 

La actualización de los parámetros de las capas ocultas también se realiza en función de cómo contribuyen al error.

Por ejemplo, para los pesos \(w_{11}\), \(w_{21}\), \(b_1\), los cálculos serían:

\[\frac{\partial \text{Loss}}{\partial w_{11}} = \frac{\partial \text{Loss}}{\partial a_{\text{salida}}} \cdot \frac{\partial a_{\text{salida}}}{\partial z_{\text{salida}}} \cdot \frac{\partial z_{\text{salida}}}{\partial a_1} \cdot \frac{\partial a_1}{\partial z_1} \cdot \frac{\partial z_1}{\partial w_{11}}\]

Luego, utilizamos la regla de la cadena para propagar el gradiente hacia atrás y actualizar los pesos en la capa oculta:

\[w_{11} = w_{11} - \text{lr} \cdot \frac{\partial \text{Loss}}{\partial w_{11}}\]

Este proceso se repite para todos los pesos y sesgos en la red neuronal, permitiendo que la red aprenda a ajustar sus parámetros para minimizar la pérdida y mejorar su capacidad de realizar predicciones.


# Funciones de activación

En las redes neuronales, la función de activación en los perceptrones (neuronas individuales) es crucial para introducir no linealidades en el modelo y permitir que la red aprenda patrones más complejos. Aquí tienes algunas funciones de activación comunes utilizadas en perceptrones y sus escenarios de uso comunes:

1. **Función Sigmoide:**
   - *Fórmula:* \( \sigma(x) = \frac{1}{1 + e^{-x}} \)
   - *Rango:* (0, 1)
   - *Uso común:* Tradicionalmente utilizada en capas ocultas de redes neuronales para problemas de clasificación binaria. Sin embargo, ha sido reemplazada en capas intermedias por funciones más modernas como ReLU debido a algunos problemas de entrenamiento.

2. **Función Tangente Hiperbólica (tanh):**
   - *Fórmula:* \( \tanh(x) = \frac{e^{2x} - 1}{e^{2x} + 1} \)
   - *Rango:* (-1, 1)
   - *Uso común:* Similar a la sigmoide, pero con un rango que abarca valores negativos. Se utiliza en capas ocultas para problemas de clasificación y regresión.

3. **Rectified Linear Unit (ReLU):**
   - *Fórmula:* \( \text{ReLU}(x) = \max(0, x) \)
   - *Rango:* [0, +∞)
   - *Uso común:* Ampliamente utilizado en capas ocultas. Ayuda a abordar el problema de la desaparición del gradiente y acelera el entrenamiento.

4. **Leaky ReLU:**
   - *Fórmula:* \( \text{Leaky ReLU}(x) = \max(\alpha x, x) \), donde \(\alpha\) es una pequeña constante positiva.
   - *Rango:* (-∞, +∞)
   - *Uso común:* Similar a ReLU pero con una pendiente pequeña para valores negativos, lo que ayuda a evitar neuronas muertas.

5. **Unidad de Umbral Lineal (Step Function):**
   - *Fórmula:* \( \text{Step}(x) = \begin{cases} 0 & \text{si } x < 0 \\ 1 & \text{si } x \geq 0 \end{cases} \)
   - *Uso común:* Puede utilizarse en problemas de clasificación binaria simple, pero generalmente no se prefiere en capas ocultas debido a su no diferenciabilidad.

Estas funciones de activación son componentes esenciales para la operación de redes neuronales, y la elección de una función específica puede depender del problema específico que estás abordando y de la arquitectura de la red. La ReLU y sus variantes (como Leaky ReLU) son ampliamente utilizadas en la actualidad debido a su eficacia en el entrenamiento de redes profundas.

## Neuronas muertas

Se dice que una "neurona está muerta" cuando siempre produce la misma salida sin importar la entrada. En otras palabras, la activación de la neurona siempre es cero (o algún valor constante) y no contribuye al aprendizaje del modelo.

Esto suele ocurrir cuando se utiliza la función de activación ReLU (Rectified Linear Unit) y la entrada de la neurona es siempre negativa. Dado que la ReLU convierte todas las entradas negativas a cero, si una neurona tiene una salida cero para todas las entradas durante el entrenamiento y nunca se activa, se considera "muerta".

Las neuronas muertas pueden ser problemáticas porque no están contribuyendo al aprendizaje del modelo y pueden afectar negativamente la capacidad de la red para aprender patrones complejos en los datos. Para abordar este problema, se han propuesto variantes de ReLU, como Leaky ReLU, que permiten un pequeño gradiente para entradas negativas, evitando así que la neurona esté completamente "muerta".

# Algoritmos de aprendizaje

En el ámbito del aprendizaje profundo (deep learning), se utilizan varios algoritmos de aprendizaje para entrenar modelos de redes neuronales. A continuación, se mencionan algunos de los algoritmos más comunes:

1. **Descenso del Gradiente (Gradient Descent):**
   - **Descenso del Gradiente Estocástico (SGD):** Actualiza los pesos utilizando el gradiente en mini lotes.
   - **Descenso del Gradiente con Momento:** Incorpora términos de momento para suavizar las actualizaciones de peso.
   - **Descenso del Gradiente con RMSProp:** Adapta la tasa de aprendizaje según la magnitud de los gradientes.
   - **Descenso del Gradiente con Adagrad:** Ajusta la tasa de aprendizaje de forma adaptativa para cada parámetro.

2. **Algoritmos Basados en Adam:**
   - **Adam (Adaptive Moment Estimation):** Combina conceptos de momento y RMSProp para adaptar la tasa de aprendizaje y mantener un historial de momentos anteriores.

3. **Algoritmos Basados en Adadelta:**
   - **Adadelta:** Similar a RMSProp, adapta la tasa de aprendizaje, pero con una estrategia diferente para acumular gradientes.

4. **Algoritmos Basados en Adamax:**
   - **Adamax:** Variante de Adam que utiliza el infinito norm para estabilizar las actualizaciones.

## Descenso del gradiente

El descenso del gradiente es un algoritmo de optimización utilizado para minimizar una función de pérdida (loss function) al actualizar iterativamente los parámetros de un modelo. En el aprendizaje profundo, el descenso del gradiente se utiliza para actualizar los pesos de una red neuronal.

El descenso del gradiente se basa en el concepto de gradiente, que es un vector que apunta en la dirección de mayor crecimiento de una función. En el aprendizaje profundo, el gradiente se utiliza para determinar la dirección en la que se deben actualizar los pesos de la red para minimizar la función de pérdida.

El descenso del gradiente se puede utilizar para actualizar los pesos de una red neuronal utilizando el gradiente de la función de pérdida con respecto a los pesos. El gradiente se calcula utilizando el algoritmo de propagación hacia atrás (backpropagation), que calcula el gradiente de la función de pérdida con respecto a cada peso en la red.

## Adam (Adaptive Moment Estimation)

Adam es un algoritmo de optimización popular utilizado para entrenar redes neuronales. Fue propuesto por D. P. Kingma y J. Ba en su artículo "Adam: A Method for Stochastic Optimization". Adam combina conceptos de dos métodos de optimización, el descenso del gradiente estocástico (SGD) y el método de momentos.

1. **Tasa de Aprendizaje Adaptativa:**
   - Adam adapta la tasa de aprendizaje de forma individual para cada parámetro. Utiliza estimaciones de primer y segundo momento para ajustar dinámicamente la tasa de aprendizaje durante el entrenamiento.

2. **Estimaciones de Momento:**
   - Adam mantiene dos estimaciones de momento. El primer momento (media móvil) almacena el promedio ponderado de los gradientes pasados, y el segundo momento (media móvil no centrada) almacena el promedio ponderado de los cuadrados de los gradientes pasados.

3. **Corrección de Sesgo (Bias Correction):**
   - Adam realiza una corrección de sesgo para compensar el hecho de que las estimaciones de momento son inicialmente sesgadas hacia cero, especialmente durante las primeras iteraciones del entrenamiento.

4. **Fórmula de Actualización de Pesos:**
   - La fórmula general para actualizar los pesos en Adam es:
      \[ w_{t+1} = w_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t \]
   donde \(w_t\) es el peso en la iteración \(t\), \(\eta\) es la tasa de aprendizaje, \(\hat{m}_t\) es la estimación de primer momento, \(\hat{v}_t\) es la estimación de segundo momento, y \(\epsilon\) es un término pequeño para evitar la división por cero.

Adam ha demostrado ser eficaz en una variedad de tareas y se utiliza comúnmente como un algoritmo de optimización predeterminado en muchas bibliotecas de aprendizaje profundo. Es particularmente útil en problemas con grandes conjuntos de datos y dimensiones de parámetros. Sin embargo, como con cualquier algoritmo, la elección del algoritmo de optimización depende del problema específico y puede requerir ajustes de hiperparámetros según las características del modelo y de los datos.

# Funciones de pérdida

Las funciones de pérdida, también conocidas como funciones de costo o funciones objetivo, son utilizadas para medir la discrepancia entre las predicciones de un modelo y los valores reales. La elección de la función de pérdida depende del tipo de problema que estás abordando (clasificación, regresión, etc.). Aquí tienes algunas funciones de pérdida comunes y sus escenarios de uso:

### Clasificación Binaria:

1. **Entropía Cruzada Binaria (Binary Cross-Entropy):**
   - **Fórmula:** \( L(y, \hat{y}) = - (y \log(\hat{y}) + (1 - y) \log(1 - \hat{y})) \)
   - **Uso Común:** Para problemas de clasificación binaria donde la salida esperada \(y\) es 0 o 1.

2. **Entropía Cruzada Sigmoidal (Sigmoid Cross-Entropy):**
   - **Fórmula:** \( L(y, \hat{y}) = - y \log(\sigma(\hat{y})) - (1 - y) \log(1 - \sigma(\hat{y})) \), donde \(\sigma\) es la función sigmoide.
   - **Uso Común:** Similar a la entropía cruzada binaria, pero adaptada para la salida de una función sigmoide.

### Clasificación Multiclase:

3. **Entropía Cruzada Categórica (Categorical Cross-Entropy):**
   - **Fórmula:** \( L(\mathbf{y}, \hat{\mathbf{y}}) = - \sum_{i} y_i \log(\hat{y}_i) \)
   - **Uso Común:** Para problemas de clasificación con más de dos clases.

4. **Softmax y Pérdida de Logaritmo (Softmax and Logarithmic Loss):**
   - **Fórmula:** Pérdida de logaritmo negativo aplicada después de la activación softmax.
   - **Uso Común:** Similar a la entropía cruzada categórica, a menudo utilizada en conjunto con la activación softmax.

### Regresión:

5. **Error Cuadrático Medio (Mean Squared Error - MSE):**
   - **Fórmula:** \( L(y, \hat{y}) = \frac{1}{2}(y - \hat{y})^2 \)
   - **Uso Común:** Para problemas de regresión donde se busca predecir valores numéricos.

6. **Error Absoluto Medio (Mean Absolute Error - MAE):**
   - **Fórmula:** \( L(y, \hat{y}) = |y - \hat{y}| \)
   - **Uso Común:** Similar al MSE, pero utiliza la diferencia absoluta en lugar del cuadrado.

### Otros:

7. **Huber Loss:**
   - **Fórmula:** Combina la robustez del error absoluto y la suavidad del error cuadrático.
   - **Uso Común:** Útil cuando hay presencia de outliers en los datos.

8. **Pérdida de Bisagra (Hinge Loss):**
   - **Fórmula:** \( L(y, \hat{y}) = \max(0, 1 - y \cdot \hat{y}) \)
   - **Uso Común:** Empleado en máquinas de soporte vectorial para problemas de clasificación.

# Generative Adversarial Network (GAN)

### Redes Generativas Adversarias (GAN)

Una GAN es un modelo generativo que utiliza dos redes neuronales, un generador y un discriminador, que se entrenan simultáneamente en un juego competitivo.

#### Principales Componentes de una GAN:

1. **Generador:**
   - Toma muestras aleatorias de un espacio latente y las transforma en datos sintéticos.
   - Objetivo: generar datos indistinguibles de los datos reales.

2. **Discriminador:**
   - Evalúa la autenticidad de una muestra y determina si proviene del conjunto de datos real o fue generada por el generador.
   - Objetivo: distinguir entre datos reales y generados.

#### Proceso de Entrenamiento:

1. **Fase Generativa:**
   - El generador genera datos sintéticos.
   - Estos datos se presentan al discriminador junto con datos reales.

2. **Fase Discriminativa:**
   - El discriminador clasifica muestras como reales o generadas.
   - Entrenamiento para distinguir entre datos reales y generados.

3. **Back-and-Forth:**
   - El generador se entrena para engañar al discriminador.
   - El discriminador se entrena para ser preciso en distinguir entre datos reales y generados.

4. **Optimización Conjunta:**
   - Ambas redes se entrenan iterativamente.

#### Función de Pérdida:

- Término de Pérdida del Discriminador: Minimiza la diferencia entre las clasificaciones reales y generadas.
- Término de Pérdida del Generador: Maximiza la probabilidad de que las muestras generadas sean clasificadas como reales.

#### Generación de Datos Nuevos:

El generador puede generar datos sintéticos después del entrenamiento.


# Variational Autoencoder (VAE)

Un VAE es un tipo de red neuronal que combina autoencoders y modelos generativos probabilísticos.

1. **Autoencoder: Encoder y Decoder**
   - *Encoder*: Transforma los datos de entrada en una distribución en el espacio latente.
   - *Decoder*: Mapea las muestras del espacio latente de vuelta al espacio de datos original.

2. **Distribución Latente**
   - Las variables latentes siguen una distribución probabilística, comúnmente una distribución normal.

3. **Función de Pérdida**
   - Término de Reconstrucción: Mide cuán bien el modelo puede reconstruir las muestras de entrada desde las variables latentes.
   - Término de Regularización KL (Kullback-Leibler): Mide cuánto se desvían las variables latentes de la distribución normal estándar.

4. **Generación de Datos Nuevos**
   - Una vez entrenado, el VAE permite generar datos nuevos tomando muestras de la distribución latente y pasándolas a través del decodificador.

5. **Uso en Generative Modeling**
   - Entrenamiento: Se entrena el VAE utilizando datos de entrada.
   - Generación de Datos Nuevos: Después del entrenamiento, se pueden generar datos nuevos tomando muestras aleatorias del espacio latente y utilizando el decodificador.
   - Exploración del Espacio Latente: El espacio latente aprendido puede explorarse para descubrir cómo cambiar las variables latentes afecta la generación de datos.

                            ESPACIO LATENTE
    Pregunta1                   vvvvv                   Pregunta1'
    Pregunta2       P1      C.Suma              PD1     Pregunta2'
    Pregunta3       P2      C.Multiplicación    PD2     Pregunta3'
    Pregunta4       P3                          PD3     Pregunta4'
    Pregunta5                                           Pregunta5'

        --- ENCODER ------->                  --- DECODER --->

De formas que coy a tratar de conseguir un par encoder/decoder que me permita reconstruir las preguntas originales a partir de las preguntas transformadas.

    Eso es el concepto de autoencoder.... pero en nuestro caso, vamos a aplicar un autoencoder variacional.

Los autoencoders variacionales (VAE) buscan que las variables latentes sigan una distribución normal:
    Con una media (mu) y con una desviación típica (sigma)

Una vez conseguido eso:
    Creo un conjunto de valores al azar para
    C.SUMA y C.Multiplicación (partiendo de la media y la desviación típica)
Y a esos datos, les aplico el DECODER, para generar un nuevo valor... aleatorio... que encaje con los datos que he visto en el entrenamiento.

# Principal problema al generar modelos

## SOBREAJUSTE / OVERFITTING / SOBRECALENTADO

Imaginad que quiero trabajar en BBVA.
Y pregunto a los de RRHH por los billetes... cómo van los billetes aquí?

Me dicen, me voy a saltar todas las leyes de protección de datos y te voy a dar los datos de todos los empleados de BBVA.
Me sueltan todas las nóminas... junto con algunos datos adicionales de los empleados:
- Nombre, DNI, Altura, Sexo, Edad, Estudios, Años en la empresa, Salario

Quiero hacer un modelo predictivo de un salario de un empleado de BBVA... para meter mis datos y que me ofrezca una estimación.
Tengo datos de 10.000 empleados de BBVA.... qué tal haré la predicción?

Solo necesito 1 variable para obtener un poder predictivo del 100% -> DNI

Este modelo iba a dar una predicción perfecta para los que ya son empleados... por cierto... de los cuales ya conozco su salario... y no necesito hacer ninguna predicción.
Qué va a pasar cuando le meta mi DNI? RUINA !

El modelo se ha especializado tanto en los datos de entrenamiento que no es capaz de hacer predicciones fuera de esos datos.

Aprender es un proceso de descartar información que no es relevante para el problema que quiero resolver.
Al descartar información, pierdo los datos singulares de cada individuo... y me quedo con los datos que son comunes a todos los individuos.

Felipe y a Lucas, que ambos tienen 10 años de experiencia, 5 en BBVA, mismos estudios, misma altura, edad y complexión... y uno gana 50k y otro 52k.... Pues digo... si yo tengo loas mismos años de experiencia que Lucas y Felipe... y los mismos estudios... y la misma altura... y la misma edad... y la misma complexión... pues yo ganaré en torno a 51k por ahí estará la cosa.

Al calcular lo bien o mal que funciona un modelo, es importante usar datos que no sean solo los que he usado para entrenar el modelo.