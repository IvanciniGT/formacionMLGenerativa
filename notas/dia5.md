
# Modelo PLN

El objetivo es conseguir entender un texto, en un idioma.

    Ejemplo: El perro se come un filete.

## Opción 1: Montar un programa donde vayamos codificando linea a linea las reglas gramaticales de un determinado idioma.
    
        Sujeto: El perro
        Verbo: come
        Objeto: un filete

    Descartado... ya se ha intentado.
    Complejidad del problema.

## Opción 2: Machine learning -> Redes neuronales

### Problema 1: Las redes neuronales lo que hacen es echar cuentas (combinaciones de variables, calcular funciones de activación...). No entienden textos.

#### Solución 1

    Ejemplo: El can se come un filete.
             1   8     3   4  5   6
    Ejemplo: El perro se come un filete.
             1   2     3   4  5   6
    Ejemplo: El perrito se come un filete.
             1   7      3   4    5   6
    Ejemplo: El gato se come un filete.
             1   9      3   4    5   6
    El problema que tiene esto es que tengo datos categoricos / no cuantitativos.

         perrito y perro son 2 palabras que tienen un significado muy muy parecido en esas frases.
         Un 2 y un 7 no nos hacen entender que esas palabras están relacionadas (son sinónimos).

#### Solución 2: Embeddings

    Ejemplo:               Perro come filete
    Precategoría:            1    2     3

    Vector                   1    0     0
                             0    1     0
                             0    0     1
                            ...
                             0    0     0
                             50k de posiciones

    Si represento esos vectores en un espacio tridimensional, podría calcular la distancia entre ellos.
    Cuál sería? raíz de 2
    La distancia entre todos ellos, sería raíz de 2.
    Acabo de conseguir uan forma de codificar las palabras que tenga una forma de medir la distancia entre ellas.... primero no partiendo de distancias distintas sin sentido.

    Lo que voy a tratar ahora es de reorganizar esa información de forma que vaya empezando a agrupar conceptualmente palabras.

    Partimos de un vector de tamaño 3... ya que nuestro vocabulario hemos supuesto que es de tamaño 3...
    pero si tengo un lenguaje completo... tendremos un vocabulario de tamaño enorme.
    En español, podría tener más de 500k de palabras.
    Lo que podría generar vectores de tamaño 500k... además contendrían todo ceros, menos en una posición que tendrían un 1.

    Esto me supone un problema de espacio y de tiempo de computación.
    A reducir el tamaño de ese vector me puede ayudar el hacer una mejor tokenización del texto.
    - Extraigo raíces... y terminaciones de las conjugaciones de los verbos.... lo reduciré un huevo
    - El plural lo separo de la palabra, o el género

    Qué hacemos ahora?
    Es reducir más aún el tamaño de ese vector.... intentando identificar el significado de esas palabras... y de alguna forma irlas agrupando.

                    4 dim                3 dim
        dog         0   0   0   0  1      1   0     1
        perro       1   0   0   0  0 ---> 1   0     0
        perrito     0   1   0   0  0      1   1     0
        pequeño     0   0   1   0  0      0   1     0
        chiquito    0   0   0   1  0      0   1     0
                                            ^ Tamaño pequeño <<< Concepto / PRIMERA FASE DE APRENDIZAJE
                                        ^
                                        Perro
        banco --->                        0   0     0    1    0   0   1   0   0   0   0   0   0   1
            3 conceptos: Sitio donde sentarse, empresa de billetes, grupo de peces

    Cómo se consigue esto? -> Redes neuronales

        X1      XA
        X2      XB          Paso de n dimensiones a C dimensiones
        X3      XC
        Xn

        El perro se ladra.
        El can se ladra.
        El perro se come un filete.
        El gato se come un filete.
        El gato maulla.
        El felino maulla.
        Juan es un niño.
        El niño se come un filete.
        El niño se come un pescado.
        El niño de some una manzana.
        La niña se come un filete.
        La niña se come un pescado.
        La niña de some una manzana.
        El niño se sienta en un banco
        El hombre va a un banco

        Qué conclusión puedo sacar? Que el perro, el gato, el hombre y el niño comen filetes... se parecen en eso.

        Hay distintas redes ya creadas para hacer este trabajo: word2vec, glove, fasttext

    Estos son conceptos de para lo que se usa esa palabra... Pero una palabra puede usarse para muchas cosas.
    Y en un momento dado, en función del contexto concreto en el que parece la palabra tenderá más a un significado que a otro.

    Podríamos entender el resultado de esta fase de embedding como la generación de un diccionario.
    No tenemos significados, pero tenemos relaciones entre palabras.

    Cuando tomamos como partida un modelo como BERT, GPT, esta ya lo han hecho por nosotros.
    Y se han entrenado con miles de millones de datos.

### Problema 2:

Ya conozco los "significados potenciales" de cada palabra.
Cuando quiero usar ese diccionario para entender el significado de una frase:
- Depende en que lugar de la frase aparezca la palabra tendrá un significado u otro.
    El perro se come un filete
    El león se come un perro
- Depende el contexto (el resto de palabras en la frase), tendrá un significado u otro.
    Me siento en el banco
    Saco dinero del banco

#### Solución 1:

El primer enfoque que se dió para resolver este problema fue: RRN

Vamos procesando palabra a palabra:  El perro se come un filete
                                    ---->
Las RRN iban procesando palabra a palabra. Y la idea es que la salida de una palabra, se usaba como input adicional para la siguiente palabra

    El      perro       se      come        un      filete
    v        v          v        v          v         v
    Red     Red         Red     Red         Red     Red
    v     /  v       /  v      /  v       /  v      /  v
    Salida1  Salida2    Salida3   Salida4   Salida5     Salida6

    El problema es que según la información se va retroalimentando en la red... pierde intensidad.
    Dicho de otra forma, la RED se va olvidando de los términos más antiguos de la frase.

    De alguna forma nos inventamos las LSTM (Long Short Term Memory)... que mejoraban un poco la capacidad retentiva de la red.... pero con el mismo problema (que aparecía más tarde... pero el mismo problema)

#### Solución 2: Transformers -> 

Modelo que crea google 2016-2017.
Principalmente para traducir textos... para su traductor.
Se cambia en el enfoque de las redes neuronales.... se abandonan las redes recurrentes.
Dentro de la arquitectura transformer no se usan RRN.

Este modelo es el que luego se usa en : BERT, GPT, T5, XLNet, RoBERTa, ALBERT, ELECTRA, DistilBERT, CamemBERT, XLM, CTRL, Reformer, Longformer, FlauBERT, Bart, MBART, Pegasus, Marian, ProphetNet, Speech2Text, LayoutLM, DeBERTa, Funnel Transformer, MPNet, MobileBERT, LED, BORT, Tapas, BigBird, ViT, LXMERT, UNI

Ha ido más allá de el PLN... y hoy se usan para otro tipo de situaciones (secuencias de eventos)

"Attention"

En las redes de transformers las frases no se engullen de forma secuencia... sino que se engullen de golpe.

    Las palabras se procesan todas a la vez


           El         perro       se     come        un      filete
           ------ 
           |0.56|     0.78       0.01    0.14        0.78    0.01
           |0.78|     0.56       0.01    0.14        0.78    0.01
           |0.01|     0.01       0.56    0.14        0.78    0.01
           |....|     ....       ....    ....        ....    ....
           |0.14|     0.14       0.14    0.56        0.14    0.14
           ------
            ^^^
            Embedding: Que capturan el significado genérico de la palabra (DICCIONARIO)
            E1         E2         E3        E4        E5      E6  

    Se entrenan 2 redes neuronales en paralelo

        E1, E2, E3, E4, E5, E6  ----> RED1 ----> Vectores Query: Q1, Q2, Q3, Q4, Q5, Q6 \
                                ----> RED2 ----> Vectores Key:   K1, K2, K3, K4, K5, K6  >      1 HEAD
                                ----> RED3 ----> Vectores Value: V1, V2, V3, V4, V5, V6 / 

#####  ATENCION <<< Esta es la clave de los transformers y porque funcionan tan bien!

    Como es de relevante cada palabra que parece en el contexto para cada palabra que aparece en el contexto.

           El         perro     de      Juan       se     come        un      filete    mientras    juega   al  ajedrez
           ***         0.8      0.1     0.2        0.1    0.1         0.1     0.2     < ATENCION EL
           0.5         ***      0.4     0.35       0.2    0.7         0.1     0.9     < ATENCION PERRO
           0.3         0.8      ***     0.6        0.2    0.3         0.1     0.2     < ATENCION DE
           0.1         0.5      0.6     ***        0.1    0.1         0.1     0.1     < ATENCION JUAN
           0.1         0.6      0.1     0.1        ***    0.8         0.2     0.3     < ATENCION SE
           0.1         0.9      0.1     0.1        0.1    ***         0.1     0.7     < ATENCION COME

    DA LUGAR A UNA MATRIZ DE ATENCION

           Juan       se     come        un      filete

           ¿Quién se come el filete?
            perro ---> come
            Juan  ---> come

    La red neuronal RED1 (query) se encarga de resaltar aquellos valores del Embedding 
    que hablan de lo que esa palabra necesita a su alrededor para tener sentido.

        De la palabra perro: Esta palabra necesita que a su alrededor haya un verbo para darle un sentido
        Qué necesita está palabra dentro de la frase para tener sentido

    La red neuronal RED2 (keys) se encarga de resaltar aquellos valores del Embedding 
    que hablan de lo que esa palabra ofrece a su alrededor
        De la palabra come: Esta palabra es un verbo dentro de una frase ->> ACCION
        La función que esa palabra hace dentro de la frase

    GENERA EL CONCEPTO DE ATENCION -> Otro vector

##### Refino el significado:

    La red neuronal RED3 (value) se encarga de resaltar aquellos valores del Embedding 
    más representativos en el contexto de la frase
        De la palabra come: En este contexto significa la acción de comer (engullir alimento)
                ^^^^^
                Refino el significado de la palabra. Del diccionario, me quedo con 1 significado


        Embedding de la palabra banco
        ( 0.7, 0.5, 0.03, 0.098, 0.7, ...., 0.03 ) 500 dimensiones... 5000 dimensiones
        ( 0.9, 0.02, 0.02, 0.098, 0.3)
        ^
        Esta posición de embedding, alude conceptualmente a un sitio donde sentarse
        Ya que veo que en palabras tipo: Silla, sillón, sofa, asiento, este valor también es alto


        |
        | x FILETE
        |     x COME(2)
        |       ^ x COME(1)
        |
        |             x AJEDREZ
        |-----------------------

    Se junta el vector de valor, con el vector de atención, para dar más valor a aquellos valores del vector de valor que son más relevantes para la palabra que estamos procesando.

### Aun nos queda un problema por resolver:

    El         perro     de      mi     amigo       se     come        un      filete
    El         amigo     de      mi     perro       se     come        un      filete

Si se procesan las palabras sin tener en cuenta el orden en el que aparecen, la interpretación de la frase puede ser errónea.
Este problema no lo tenían la s RRN... ya que procesaban los tokens en orden.

Es necesario aportar esa información

           El         perro       se     come        un      filete
           ------ 
           |0.56|     0.78       0.01    0.14        0.78    0.01+6
           |0.78|     0.56       0.01    0.14        0.78    0.01
           |0.01|     0.01       0.56    0.14        0.78    0.01
           |....|     ....       ....    ....        ....    ....
           |0.14|     0.14       0.14    0.56        0.14    0.14
           ------
            ^^^
            Embedding: Que capturan el significado genérico de la palabra (DICCIONARIO)
            E1 +Pos   E2 +Pos   E3 +Pos   E4 +Pos   E5 +Pos E6 +Pos

OPCION 1: El problema es que la información del embedding se diluye al sumarla con valores muchos más altos (posiciones finales de la frase)
            (1,1,1,1,1,1)
                    (2,2,2,2,2,2)
                                (3,3,3,3,3,3)
                                        (4,4,4,4,4,4)
                                                    (5,5,5,5,5,5)
                                                            (6,6,6,6,6,6)

OPCION 2: 
            (0.16, 0,16, 0,16, 0,16, 0,16)
                    (0.32, 0,32, 0,32, 0,32, 0,32)
                                (0.48, 0,48, 0,48, 0,48, 0,48)
                                        (0.64, 0,64, 0,64, 0,64, 0,64)
                                                    (0.80, 0,80, 0,80, 0,80, 0,80)
                                                            (0.96, 0,96, 0,96, 0,96, 0,96)

Al haber relativizado la posición al tamaño de la frase, los valores pierden la capacidad de hacer restas, calcular diferencias

            El niño es guapo
            1   2    3   4
                                Cuántas palabras de diferencia hay entre guapo y niño? Cuánto se separan? Separadas por 1 palabra entre medias
                                    4-2-1 = 1 palabra
                                Y entre El y guapo: 4-1-1 = 2 palabras
            El niño de mi amigo es guapo
                                Entre niño y guapo hay 4 palabras de diferencia

    Ese dato que uso es comparable entre frases. En la segunda frase las palabras guapo y niño están más separadas que en la primera frase.

            El      niño    es      guapo
            0.25    0.5     0.75    1           Tengo una diferencia de 0.5 entre la posición del niño y la posición de guapo

            El      niño    de      mi      amigo   es      guapo  que  te  cagas
            0.10    0.20    0.30    0.40    0.50    0.60    0.70    0.80  0.90  1
                                                Tengo una diferencia de 0.5 entre la posición del niño y la posición de guapo

            Cómo comparo el 0.5 de antes y el 0.5 de ahora... Por que los números son iguales... Están igual de cerca?

                            VECTOR DEL MISMO TAMAÑO QUE EL EMBEDDING (500) Pongo el número en binario
            El              0 ....  0   0   0   0   0   0
            niño            0 ....  0   0   0   0   0   1
            de              0 ....  0   0   0   0   1   0
            mi              0 ....  0   0   0   0   1   1
            amigo           0 ....  0   0   0   1   0   0
            es              0 ....  0   0   0   1   0   1
            guapo           0 ....  0   0   0   1   1   0
            que             0 ....  0   0   0   1   1   1
            te              0 ....  0   0   1   0   0   0
            cagas           0 ....  0   0   1   0   0   1
            por             0 ....  0   0   1   0   1   0
            las             0 ....  0   0   1   0   1   1
            noches          0 ....  0   0   1   1   0   0
            de              0 ....  0   0   1   1   0   1
            primavera       0 ....  0   0   1   1   1   0
            en              0 ....  0   0   1   1   1   1
            el              0 ....  0   1   0   0   0   0
            campo           0 ....  0   1   0   0   0   1

                                                        ^ PATRON: 0 1 0 1 0 1 0 1 0 1
                                                    ^     PATRON: 0 0 1 1 0 0 1 1 0 0
                                                ^         PATRON: 0 0 0 0 1 1 1 1 0 0 0 0 1 1 1 1 0 0 0 0
                                            ^             PATRON: 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0    
            Eso se podría representar como una función sinusoide
                P E(pos,2i) = sin(pos/100002i/dmodel)
                P E(pos,2i+1) = cos(pos/100002i/dmodel)
            
            El sin o cos está entre -1 y 1... estamos en el mismo orden de magnitud que los valores de los embeddings

### Estructura Multi-head Attention

Lo que se hace es ese mismo proceso de ponderar los significados (embeddings) con la atención, usando distintas formas de determinar dónde se pone la atención.

Cada head se especializa en poner más énfasis en una determinada porción de la frase

    FRASE: El perro de Juan se come un filete mientras juega al ajedrez

    HEAD 1:
        El perro de Juan se come un                 filete mientras juega al ajedrez

    HEAD 2:
        El perro de Juan                            se come un filete mientras juega al ajedrez

    HEAD 3:
        El perro de Juan se come un filete          mientras juega al ajedrez

Lo que se pretende con ésto es buscar relaciones entre trozos de la frase con trozos de la frase.

De alguna forma, ésto es lo que va realizando un análisis sintáctico en profundidad dentro de la frase

Alguna de esas relaciones será más intensa que otra... y hace la frase tenga más sentido
    Acabo de separar conceptualmente el Sujeto del predicado!

Se calculan todas esas y luego se consolida la información dando más peso a aquellas cabezas que hayan mostrado relaciones de mayor intensidad

El resultado de este trabajo, se junta de nuevo con LA ENTRADA ORIGINAL

Y vuelvo a hacer TODO DESDE EL PRINCIPIO.... muchas veces.... montones de veces

Intento conseguir un entendimiento profundo de las relaciones que hay entre las palabras que aparecen en esa frase.

Y ESTA ES LA PRIMERA PARTE DE LOS TRANSFORMERS: ENCODER


# Parámetros

Red tiene:
CAPA ENTRADA DE 10000 DATOS
CAPA INTERMEDIA CON 5000 PERCEPTRONES
SEGUNDA CAPA DE 2000 PERCEPTRONES
CAPA SALIDA DE 100 DATOS

10000                   * 5000 + 5000 = PARAMETROS A CALCULAR EN CAPA 1 = 50.005.000
^ PESOS de cada entrada           ^segsos (bias) de cada perceptron

5000 * 2000 + 2000 = PARAMETROS A CALCULAR EN CAPA 2 = 10.002.000
2000 * 100 + 100 = PARAMETROS A CALCULAR EN CAPA 3 = 200.100

TOTAL parámetros en el modelo a calcular = 60.207.100 de parámetros a calcular

# Modelo BERT

Se queda con la estructura de encoder de los transformers...
Acabando con una salida que refleja el significado de la frase.

Posteriormente esa salida se usa para diferentes tareas... le vamos poniendo distintos DECODERS

    - Clasificación de texto
    - Preguntas y respuestas
    - Identificar la frase siguiente

Ese decoder es el que nosotros posteriormente ENTRENAMOS cuando usamos un modelo BERT

En el caso de bert nos quedamos solo con el concepto de ENCODER que se define en el modelo TRANSFORER
para generar una salida que es el vector con el significado de la frase:
(0.928, 0.23, 0.98,....)
Qué significado le doy a esos números? NI IDEA

Claro, si no tengo ni idea de lo que significan, cómo se si el modelo está generando algo coherente o no?
Cómo soy capaz de entrenar el modelo para que cada vez genere algo más coherente?

Esto fue un problema. 
Se añadieron redes por detrás para entrenar el modelo:
- Se añadió una red neuronal que se encargaba de predecir la siguiente frase
- Se añadió otra red que enmascaraba palabras de una frase (15%) y se le pedía al modelo que las rellenara
Eso son cosas que si podía validar que se fueran realizando correctamente.

Posteriormente lo que hacemos , cuando usamos el modelo BERT es añadir una nueva red neuronal por encima, que tome la información que genera BERT (~el significado de la frase) y que entrenamos en tareas específicas.

Ese entrenamiento es muchísimo menos costoso que el entrenamiento de BERT... ya que BERT es gigantesco.... y lleva mucho tiempo entrenarlo.

## Modelo GPT

A diferencia de BERT que se queda con la parte de ENCODER de los transformers... GPT se queda con la parte de DECODER de los transformers.