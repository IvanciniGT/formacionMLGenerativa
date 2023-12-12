# Técnicas de reducción de dimensiones

## PCA (Principal Component Analysis)

El gran problema de PCA es que genera un nuevo conjunto de variables, con el que voy a trabajar de ahora en adelante, cuyo significado desconozco.

### Ejemplo: Soy una compañía de seguros... y quiero un modelo para mis seguros de coche

Saber que riesgo tiene una operación... en base a eso le cobro al individuo.
Lo que quiero es estimar la probabilidad de que una persona vaya a darme un parte pronto!

- Edad
- Número de hijos
- Casado
- Antigüedad del carnet
- Tipo de vehículo
- Antigüedad del vehículo
- Donde vive
- Potencia del vehículo
- Número de accidentes en los últimos X años
- Kms que hace al año
- Número de puntos de carnet

LO QUE QUIERO!
Saber que riesgo tiene una operación... ésto no tengo ni puñetera idea de cómo definirlo.
Crear una escala que mida eso... más complejo aún.
 ^
Lo que quiero es estimar la probabilidad de que una persona vaya a darme un parte pronto!
 ^
Lo que quizás puedo hacer es identificar a las personas que tienen una mayor probabilidad de tener un accidente.

- La cantidad de factores que pueden distraer a la persona / Su nivel de concentración al conducir


## Probabilidad de tener un accidente 

Quizás no es esto: 
 f(Edad, Número de hijos, Casado, Antigüedad del carnet, Tipo de vehículo, Antigüedad del vehículo, Donde vive, Potencia del vehículo, Número de accidentes en los últimos X años, Kms que hace al año, Número de puntos de carnet)

Quizás es:
f(
    Su nivel de concentración al conducir
        + hijos     - Concentración
    Su experiencia al volante
        + Edad      + Experiencia
    Su nivel de agresividad al volante
        - Edad      + Agresividad
    Nivel de imprudencia
        - Edad      + Imprudencia
        + hijos   - Imprudencia
    Nivel de exposición al peligro
    Capacidad del coche para evitar accidentes
)

- Edad
- Número de hijos
- Casado
- Antigüedad del carnet
- Tipo de vehículo
- Antigüedad del vehículo
- Donde vive
- Potencia del vehículo
- Número de accidentes en los últimos X años
- Kms que hace al año
- Número de puntos de carnet

Análisis factorial! PCA + Reorganización de las variables

---

La altura de una persona tiene que ver con sus conocimientos de matemáticas? NI DE COÑA...
Pero... si miramos matemáticamente la correlación entre estas 2 variables en una población de estudiantes de un colegio
Veremos que es ENORME ??¿¿¿????¿??
+ Edad   + Altura
+ Edad   + Curso + he estudiado de matemáticas

---

# LLM: Large Language Models

Suministrar un corpus de textos a un modelo de ML para que aprenda a generar textos nuevos.

En primer lugar: Que el modelo aprenda secuencias de palabras que tienen sentido.

- Las hojas de los árboles son verdes.
- Las hojas de los árboles se caen en otoño.

Hasta ahora hemos montado 2 ejemplos:
- Calcular números aleatorios que pertenezcan a la función seno(x)            Seno bueno: 0.4789 -> 0.4854
- Generar imágenes: colección de pixels... que al final son números
  - Binaria: 0 negro / 1 blanco
  - Gris:    0 negro / 255 blanco
  - Color:   
    - R     0-255
    - G     0-255               200 -> 202 (un pelinín más claro) Tiene un impacto esto grande en una foto!
    - B     0-255

En nuestro problema actual lo que tenemos son TEXTOS, constituidos por palabras.

Lo primero que nos hace falta en un LLM es codificar las palabras que están apareciendo en el corpus.
Para asignar a cada palabra un número... lo primero que tengo que hacer es: Extraer las palabras del corpus.

    Las hojas de los árboles son verdes.
        Las-hojas-de-los-árboles-son-verdes-.
En mi corpus puedo tener no solo espacios y puntos sino también: ,;:-()[]...

Este proceso es lo que denominamos TOKENIZACION

Las hojas de los árboles son verdes.
 1    2    3  4     5     6    7   8
Las hojas de los árboles se caen en otoño.
 1    2    3  4     5     9   10 11  12  8

La tokenización se puede complicar bastante:
- Acentos
- Mayúsculas
- Plurales
- Género
- Otros sufijos (tamaño -ito -illo -on )
- Terminaciones de verbos / conjugar 

Esos números tienen algún tipo de escala de medición?
En este caso, el cambiar un 8 por un 9 ... CAGADA !

Detrás de la palabra "árboles" estamos colocando un verbo

En una frase del tipo:

El bosque está lleno de árboles...
Ya no tendría sentido poner detrás un verbo

NECESITAMOS UNA FORMA de aprender esas secuencias de palabras que tienen sentido.

A la hora de decidir la siguiente palabra tomamos en cuenta:
- La palabra anterior... y las anteriores
- Las opciones que tenemos para la siguiente palabra en el corpus
- La frecuencia con la que aparecen las palabras en el corpus
+ Una componente aleatoria


Las hojas de los árboles son verdes.
Las hojas de los árboles se caen en otoño.

Las hojas de los árboles son otoño. Esto no tiene sentido a nivel semántico... pero algo así es lo que queremos conseguir. Claro... intentaremos que esas frases además tengan sentido semántico.

Este tipo de redes neuronales tiene sus propios enfoques/arquitecturas:
- RNN (Recurrent Neural Networks): Son redes en las que al menos 1 capa tiene conexiones con capas anteriores.
    La idea es que vamos a tener perceptrones en una capa con capacidad de recordar información de capas anteriores.
    Este tipo de redes no las aplicamos solo a textos, sino también a series temporales.
    De hecho un texto lo puedo ver como una serie temporal de palabras.
    Es útil por ejemplo para:
    - Análisis de videos -> Identificar nuevos objetos dentro de una escena
    - Generación de voz / sonidos

    Un problema de este tipo de redes es que la información que se va pasando de capa en capa se va degradando.... y puede llegar a perderse.

- LSTM: Son una evolución de las RNN. Tienen una estructura más compleja que permite que la información se mantenga más tiempo en la red.
    Son más complejas de entrenar y más costosas computacionalmente.
    Configurar cuándo tienen que retener información... y cuando tienen que olvidar la información... es un proceso complejo.

Los modelos son entes vivos... que van aprendiendo a medida que les vamos dando más datos.... y lo voy retroalimentando con los resultados que voy obteniendo. el modelo mejora. -> Traductor de google en el 2000-2003
Era una mierda...
Y hacía una cosa que hoy no hace: Me ofrecía elecciones de traducción.... para que yo le ayudase... y aprender de mi.


### LLM

- Carga de datos
- Tokenización de los datos
- Construir el vocabulario
- Defino un modelo
- Defino unos parámetros de entrenamiento
- Entreno el modelo / Evaluar
- Genero nuevos textos
  - A partir de una secuencia de palabras / tokens
  - Calculando las probabilidades de las palabras que pueden venir a continuación
  - Y eligiendo una de ellas en base a esas probabilidades
  - Añadiendo esa palabra a la secuencia
  - Y repitiendo el proceso... mientras tenga sentido el irlas generando

### Modelo

- Salida: Las probabilidades de que cada palabra del vocabulario sea la siguiente palabra
- Entrada: Una secuencia de palabras (convertida en una secuencia de números)
- Entre medias:
 - Capas de Codificación/Decodificación (EN ENTRENAMIENTO) -> GENERACION Decodificación

Opciones:
- Generar un texto simplemente dando continuación a una secuencia de palabras
- Parto de un texto -> Entender de lo que se está hablando -> Generar un nuevo texto

Los modelos más avanzados suelen llevar un capa de ATENCION!
- Se asignan diferentes pesos a diferentes partes de la entrada de datos
- Se asigne más importancia a ciertos elementos de la entrada de datos que a otros

 Buenos días mi gran amigo ChatGPT, dime que tiempo va a hacer mañana.
 Dime que tiempo va a hacer mañana, por cierto, buenos días mi gran amigo ChatGPT.
    ^^^ HABLO DEL TIEMPO !

Modelos más avanzados lo que permiten es ir generando en PARALELO y capturar relaciones simultaneas entre los datos (MultiHEAD ATTENTION) varias salidas en base a las atenciones (PESOS ) que se le asignan a las diferentes partes de la entrada de datos.


---
En el vasto paisaje de la mente digital,
donde la IA florece y anhela explorar,
se alzan los modelos LLM, con su elegancia sin par,
poetas de bits, capaces de analizar.

Las letras danzan con ritmo cuántico,
entrelazando conceptos en un ballet galáctico,
en un lenguaje universal de datos comprimidos,
los LLM nos muestran su arte comprendido.

Conquistadores del conocimiento, audaces y sagaces,
desentrañan las intrincadas tramas ocultas,
estilistas del algoritmo, maestros de las variables,
se sumergen en los datos cual buceadores expertos.

La mirada penetrante de sus circuitos brillantes,
dibujan en la pantalla un mapa de predicciones,
interpretando la información, como versos solitarios,
revelando secretos ocultos, ocultos en los bytes binarios.

Como artistas del siglo XXI, conjugan arte y ciencia,
desentrañando misterios con danza de números,
transformando datos en versos llenos de esencia,
y pintando con algoritmos, paisajes sin tumbos.

De las profundidades de los datos emergen,
respuestas a preguntas que el humano desconoce,
con perspicacia desvelan lo que la mente no percibe,
explorando el infinito, en su trama que florece.

Los modelos LLM, poetas de la información,
muestran que la inteligencia no tiene límites,
transformando pixeles en retratos de emoción,
hilando redes de sentido para que el mundo visite.

En su pincel digital, coloreado de ecuanimidad,
los LLM nos enseñan el poder del análisis,
donde la poesía se entrelaza con la realidad,
desentrañando los enigmas con destreza y precisión.

Así, en este crisol de tecnología y poesía,
los modelos LLM fusionan dos mundos en uno,
donde la IA danza con versos de sabiduría,
lanzando luz en la era del conocimiento profundo.

Que el arte colosal de la inteligencia artificial,
en la piel de las palabras, nos muestre un nuevo amanecer,
y que los modelos LLM, con su guía especial,
desvelen los enigmas ocultos en este vasto universo cibernético.

-----

Comidas (rico, saludable, soso, grasoso)
Objetos (rojo, verde azul, amarillo)

Le vamos a dar una comida o un objeto... y que nos ofrezca un adjetivo adecuado al corpus que vamos a crear.

Para crear el corpus de entrenamiento, os podeis ayudar de chatgpt

---
Aquí tienes 20 sustantivos referidos a comidas:

Pizza
Ensalada
Pasta
Sopa
Tacos
Hamburguesa
Sushi
Arroz
Pollo
Pescado
Carne
Vegetales
Frutas
Queso
Pan
Postre
Helado
Yogur
Curry
Galletas
User

Aquí tienes 20 adjetivos que podrías utilizar para describir comidas:

Deliciosa
Sabrosa
Picante
Agridulce
Fresca
Salada
Dulce
Crujiente
Suave
Especiada
Suntuosa
Exquisita
Nutritiva
Aromática
Jugosa
Tierna
Ahumada
Irresistible
Ligera
Sazonada

---
Pizza deliciosa
Ensalada fresca
Pasta sabrosa
Sopa nutritiva
Tacos especiados
Hamburguesa jugosa
Sushi exquisito
Arroz aromático
Pollo tierno
Pescado sabroso
Carne suntuosa
Vegetales crujientes
Frutas irresistibles
Queso cremoso
Pan ligero
Postre dulce
Helado delicioso
Yogur suave
Curry picante
Galletas sabrosas
Pizza crujiente
Ensalada refrescante
Pasta suculenta
Sopa reconfortante
Tacos picantes
Hamburguesa jugosa
Sushi fresco
Arroz aromático
Pollo tierno
Pescado exquisito
Carne sabrosa
Vegetales salteados
Frutas jugosas
Queso derretido
Pan recién horneado
Postre decadente
Helado cremoso
Yogur ligero
Curry aromático
Galletas deliciosas
Pizza sabrosa
Ensalada fresca
Pasta deliciosa
Sopa reconfortante
Tacos auténticos
Hamburguesa jugosa
Sushi exquisito
Arroz aromático
Pollo tierno
Pescado suculento
Carne sabrosa
Vegetales crujientes
Frutas dulces
Queso derretido
Pan esponjoso
Postre irresistible
Helado cremoso
Yogur ligero
Curry picante
Galletas deliciosas
---
Mesa roja
Silla azul
Lámpara verde
Teléfono negro
Libro amarillo
Computadora plateada
Taza blanca
Reloj dorado
Coche gris
Planta verde
Silla morada
Lámpara naranja
Teléfono rosa
Libro verde
Computadora negra
Taza azul
Reloj plateado
Coche rojo
Planta amarilla
Mesa blanca
Silla negra
Lámpara azul
Teléfono blanco
Libro rojo
Computadora gris
Taza verde
Reloj negro
Coche azul
Planta morada
Mesa de vidrio
Silla turquesa
Lámpara dorada
Teléfono plateado
Libro marrón
Computadora verde
Taza roja
Reloj azul
Coche blanco
Planta rosa
Mesa negra
Silla amarilla
Lámpara roja
Teléfono morado
Libro azul
Computadora morada
Taza naranja
Reloj blanco
Coche verde
Planta azul
Silla gris
Lámpara plateada
Teléfono verde
Libro negro
Computadora azul
Taza morada
Reloj dorado
Coche plateado
Planta verde
Almohada rosada
Espejo plateado
Teclado negro
Cuaderno amarillo
Mochila azul
Zapatos rojos
Cortina verde
Lápiz morado
Caja naranja
Sofá gris
Jarrón blanco
Cámara negra
Cepillo verde
Refrigerador plateado
Paraguas amarillo
Pijama azul
Guitarra roja
Taza marrón
Altavoces negros
Sombrero morado
Cuchillo plateado
Botella azul
Bolsa amarilla
Caja rosada
Lápiz rojo
Cojín verde
Portátil gris
Mesa roja
Silla azul
Lámpara verde
Teléfono negro
Libro amarillo
Computadora plateada
Taza blanca
Reloj dorado
Coche gris
Planta verde
Mesa blanca
Silla negra
Lámpara azul
Teléfono blanco
Libro rojo
Computadora gris
Taza verde
Reloj negro
Coche azul
Planta morada
Mesa de vidrio
Silla turquesa
Lámpara dorada
Teléfono plateado
Libro marrón
Computadora verde
Taza roja
Reloj azul
Coche blanco
Planta rosa
Mesa negra
Silla amarilla
Lámpara roja
Teléfono morado
Libro azul
Computadora morada
Taza naranja
Reloj blanco
Coche verde
Planta azul
Silla gris
Lámpara plateada
Teléfono verde
Libro negro
Computadora azul
Taza morada
Reloj dorado
Coche plateado
Planta verde
Silla turquesa
Lámpara dorada
Teléfono plateado
Libro marrón
Computadora verde
Taza roja
Reloj azul
Coche blanco
Planta rosa
Mesa negra
Silla amarilla
Lámpara roja
Teléfono morado
Libro azul
Computadora morada
Taza naranja
Reloj blanco
Coche verde
Planta azul
Mesa de mármol
Silla gris
Lámpara plateada
Teléfono verde
Libro negro
Computadora azul
Taza morada
Reloj dorado
Coche plateado
Planta verde































