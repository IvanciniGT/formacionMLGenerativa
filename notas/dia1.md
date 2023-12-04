# Entorno para la formación

- Python

---

# Machine Learning

Le vamos a pedir a una computadora que genere (que escriba) un programa por nosotros.
Por qué? En ocasiones el programa es demasiamo complejo para nuestro entendimiento, en ocasiones, simplemente no nos merece la pena escribirlo nosotros.
En este proceso, le vamos a dar a la computadora algunas instrucciones (pistas).

Distintos tipos de programas: 
- Clasificar información
- Buscar patrones dentro de datos
- Generar nuevo contenido <<<

Ejemplo:
- Traigo a un niñ@ de 1 año. Cuánto pesa? 7 Kgs, 6 Kgs
- Traigo a un niñ@ de 10 años. Cuánto pesa? 30 Kgs, 28 Kgs

Lo que estamos dando son estimaciones. De dónde las sacamos? De nuestra experiencia. A lo largo de nuestra vida, hemos visto personas de esas edades/ y tenido esas edades... y de alguna forma extrapolamos información.

Aplicar técnicas estadísticas: Regresión.

A más datos: mejor estimación. Altura, medida cintura (o de un muslo)

## Qué información me sirve de cara a hacer estas predicciones

Quiero estimar el peso de una persona (entre 0-16 años)
3 datos que puedo usar: Altura, Edad, Sexo
¿Cuál guarda más relación con el peso?
- Altura
- Edad
- Sexo

Dado que una persona tiene una altura X, cómo de bien soy capaz de estimar su peso?
    Soy capaz de acertar un 80% de los escenarios con +-10Kgs
Dado que una persona tiene una edad de X años, cómo de bien soy capaz de estimar su peso?
    De alguna forma parecido a la altura
Dado que una persona tiene como sexo: HOMBRE, cómo de bien soy capaz de estimar su peso?
    3 Kgs hasta 70 kilos . NPI
    
El tema es que altura y edad van bastante de la mano. Dentro de nuestra estimación, decimos que edad y altura son variables con una colinealidad alta. Dicho de otra forma: LAS 2 VARIABLES ME CUENTAN MAS O MENOS LO MISMO con respecto al peso.

El sexo, guarda algún tipo de relación con la edad (ninguna) o con la altura (un poquito)
El sexo me cuenta (con respecto al peso) información que la edad o la altura no me están contando... aunque sea poca información.

## Soy profe y diseño un examen de matematicas del colegio (PRIMARIA)

- Pregunta 1: 2+9
- Pregunta 2: 10+8
- Pregunta 3: 3x7
- Pregunta 4: 8x5
- Pregunta 5: 12x7

¿Para qué quiero hacer este examen? Finalidad del examen? Medir el "nivel de matemáticas"?
El primer problema con el que me encuentro es que me cuesta definir lo que quiero medir.
El segundo problema es asignar una escala de medida.

Ese test contiene preguntas. En concreto 5 preguntas. De aguna forma, pienso que esas preguntas guardan relación con el concepto que quiero medir

Quizás lo que me intera es ordenar a mis alumnos por el conocimiento de la asignatura: SOBRESALE, QUE SE HACER NOTAR

Cuántas cosas miden estas preguntas... Si me pongo a pensar un poco en ello, como experto en matemáticas:
- Capacidad para hacer sumas                P1, P2, P5
- Capacidad para hacer multiplicaciones     P3, P4, P5

De alguna forma, podría medir esos 2 conceptos: Capacidad de Suma / Capacidad de multiplicar... Y luego usarlos para estimar los conocimientos de matemáticas de los alumnos.


Esto que acabamos de hacer aquí se denomina: Técnica de reducción de dimensiones

## En qué consiste un modelo de machine learning

Generar un programa M(entradas: E1, E2, E3... En) -> Salida

Ese programa puedo escribirlo yo... o puedo dándole unas pautas a la computadora, dejar que lo escriba ella. Además de las pautas, le daré unos datos de entrenamiento:

E1  E2  E3  E4  E5      Salida
0   0   0   0   0       0

1   1   1   1   1       10


Edad     Sexo     -->      Peso
1           H                5Kgs
1           H                5.2Kgs
1           H                5.8Kgs
1           M                5.1Kgs
1           M                6Kgs
1           M                5.9Kgs
....

Si tuviera miles de millones de datos, Podría hacer una buena estimación del peso? BUENO... podría hacer una estimación.
Mejor que si tuviera solo millones de datos? Posiblemente igual

## Sucesos deterministas vs no deterministas

### Suceso determinista

Dada una entrada (un conjunto de datos de partida), siempre obtengo el mismo resultado: 
- Si tengo un objeto con un determinado peso y con un determinado coeficiente de rozamiento con el suelo y le aplico tal fuerza durante tanto tiempo, cuánto avanza? SOY CAPAZ DE CONOCER EL RESULTADO A PRIORI
- Si pongo una mesa (normal... del ikea) en medio de una sala... y pasan 1000 millones de humanos entre 18-60 años...
  y les pregunto, qué tipo de mueble es ese? Qué me contestan todos: MESA -> Reconocimineto de imágenes

### Suceso no determinista

Dada una entrada (un conjunto de datos de partida), no siempre obtengo el mismo resultado. Por qué?
- El resultado depende de factores que no estoy teniendo en cuenta
  - Dado que un niño (varón) de 7 años mide 85 cms... puedo saber su peso? NO 
- El resultado depende de factores aleatorios (Es subjetivo)
  - Es bonita la mesa? SI / NO

Por muchos datos que tenga, no siempre voy a capaz de hacer una buena estimación (son sucesos no deterministas)

---

# Cuando creamos modelos predictivos (clasificación)

Este cliente me va a pagar la hipoteca o no?

Esto es un proceso determinista o no determinista? NO DETERMINISTA
De qué dependerá que este hombre/mujer vaya a pagar la hipoteca? 
    - Dinero que tiene la familia
    - Nivel educativo
    - Tipo de contrato que tenga
    - De si dentro de 20 años va a ganar la bonoloto
    - De si dentro de 20 años le va a pegar un infarto -> AZAR
    - De si dentro de 10 minutos le pilla un coche -> IMPOSIBLE DE PREDECIR

# Modelos generadores de contenido

Estos modelos se basan en sucesos deterministas o no deterministas?
Van a ser fenómenos no deterministas... pero ... el que está generando el conjunto de datos es la computadora... que no tiene libre albedrío:
La computadora, si le pido que represente un "1" me decidirá la inclinación de esa representación que va a generar:  | / \

Le pido a la computadora que salude a una persona: "Buenos días"; "Hola"; "Qué tal?"
Sería adecuado decir "buenos tal"? NO
Sería adecuado decir "Qué días"? NO
Sería adecuado decir "Hola días"? NO

Pero una cosa será continuar un texto: Dado que tenemos escrito: "Qué" y estamos haciendo un saludo, qué palabra es la más idonea para poner ahí de entre las palabras que tengo? Qué tal?
ESTO Es una cosa... pero otra distinta es: Elegir si comienzo por: Qué, Hola, o Buenos... Y ahí vamos a tener una componente aleatoria, que tendremos que introducir en estos programas.

Estas fotos de números (dígitos) junto con el dígito que representan:
imagen00001.png -> 0
imagen00002.png -> 1
imagen00003.png -> 4
imagen00004.png -> 5
imagen99999.png -> 9
Y luego te pido que habiéndo estudiado esas imágenes y sabiendo el dígito que representaban, me estimes el dígito que representa una nueva imagen. -> Modelo predictivo (clasificación)

En lugar de eso, le puedo decir: habiéndo estudiado esas imágenes y sabiendo el dígito que representaban, generame una nueva imagen para representar el 9. -> Modelo generativo

Lo que qiuero es generar una colección nueva de datos que produzcan una foto que represente el dígito 9.