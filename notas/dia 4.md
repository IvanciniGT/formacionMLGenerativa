# Diferencias entre la Inteligencia Artificial Generativa y los Modelos de Lenguaje de Gran Tamaño

## Inteligencia Artificial Generativa (IA Generativa)

**Definición y Enfoque:** La IA Generativa se refiere a un conjunto de algoritmos de IA diseñados para crear contenidos nuevos y originales. Estos pueden incluir imágenes, música, texto, videos, etc. Se basan en el aprendizaje de patrones y estructuras de datos existentes para generar nuevos datos que sean similares en estilo o contenido.

**Técnicas Utilizadas:** Incluyen redes generativas adversarias (GANs), autoencoders variacionales y otros métodos basados en redes neuronales profundas. Estas técnicas aprenden a modelar y replicar la distribución de los datos de entrada.

**Aplicaciones:** Creación de arte y diseño, generación de música, síntesis de voz, entre otros. Son especialmente útiles en la creación de contenido que requiere un alto grado de originalidad y creatividad.

**Ejemplos Notables:** DeepDream de Google, GANs utilizadas para crear arte o para el diseño de moda, sistemas de generación de música como Jukedeck.

## Modelos de Lenguaje de Gran Tamaño (LLM)

**Definición y Enfoque:** Los LLM son modelos de inteligencia artificial específicamente diseñados para entender, interpretar, generar y manipular el lenguaje humano. Estos modelos son entrenados con grandes cantidades de texto para aprender estructuras lingüísticas, gramática y contextos.

**Técnicas Utilizadas:** Se basan en arquitecturas de redes neuronales como los Transformadores, que son eficaces para procesar secuencias de texto y aprender relaciones complejas en los datos.

**Aplicaciones:** Traducción automática, asistentes virtuales, generación de respuestas en chatbots, resumen de textos, generación de contenido escrito y más.

**Ejemplos Notables:** GPT-3 y GPT-4 de OpenAI, BERT de Google, T5 (Text-To-Text Transfer Transformer).

## Diferencias Clave

- **Objetivo y Salida:** Mientras que la IA Generativa se enfoca en crear datos nuevos en diversos formatos (no solo texto), los LLM están especializados en tareas relacionadas con el procesamiento y generación de lenguaje natural.

- **Técnicas y Modelado:** Las técnicas en IA Generativa suelen ser más variadas y enfocadas en la simulación de la creatividad humana. Por otro lado, los LLM utilizan modelos avanzados de procesamiento de lenguaje natural.

- **Aplicaciones Prácticas:** La IA Generativa tiene un rango de aplicaciones más amplio en términos de tipos de contenido (como imágenes, sonidos, etc.), mientras que los LLM están limitados al ámbito del lenguaje y la comunicación.

Aunque ambos caen bajo la categoría de la inteligencia artificial y utilizan técnicas avanzadas de aprendizaje automático, sus enfoques, aplicaciones y tipos de salida son distintos.

---
# Hugging Face

Hugging Face es una empresa de tecnología centrada en el procesamiento del lenguaje natural (NLP) y la inteligencia artificial (IA). Es conocida principalmente por su trabajo en modelos de lenguaje de última generación y por proporcionar una plataforma y comunidad para la investigación y desarrollo en IA.

## Biblioteca Transformers

La biblioteca "Transformers" es un producto destacado de Hugging Face. Es una biblioteca de código abierto que proporciona:

- **Implementaciones de Modelos de IA de Última Generación:** Incluye una gran variedad de modelos preentrenados para NLP, como BERT, GPT-2, GPT-3, T5, DistilBERT, y muchos más. Estos modelos han sido entrenados en conjuntos de datos extensos y están optimizados para diversas tareas de NLP.

- **Facilidad de Uso:** Permite a los desarrolladores e investigadores trabajar con estos modelos de forma sencilla, reduciendo la complejidad de la implementación y el uso de técnicas avanzadas de IA.

- **Interoperabilidad:** Es compatible con los frameworks de aprendizaje profundo más populares, como PyTorch y TensorFlow, lo que facilita su integración en proyectos existentes.

- **Flexibilidad para la Investigación y Desarrollo:** La biblioteca es ampliamente utilizada tanto en la investigación académica como en aplicaciones industriales, debido a su flexibilidad y capacidad para personalizar y afinar los modelos para tareas específicas.

## "Models" en la Web de Hugging Face

En la sección "Models" del sitio web de Hugging Face, encuentras una extensa colección de modelos preentrenados proporcionados por la comunidad y la empresa. Estos modelos cubren una amplia gama de aplicaciones de NLP y pueden ser utilizados para:

- **Tareas Específicas de NLP:** Esto incluye traducción automática, generación de texto, análisis de sentimientos, respuesta a preguntas, resumen de texto, etc.

- **Diversos Idiomas y Dominios:** Los modelos están disponibles para diferentes idiomas y han sido entrenados en una variedad de dominios, lo que los hace aplicables a contextos específicos.

- **Investigación y Experimentación:** Los investigadores y desarrolladores pueden descargar estos modelos, experimentar con ellos y ajustarlos a sus necesidades específicas.

- **Colaboración y Contribución de la Comunidad:** La plataforma permite que los usuarios compartan sus propios modelos entrenados, fomentando así una comunidad colaborativa.

Hugging Face y su biblioteca Transformers representan una parte importante del ecosistema de NLP y IA, proporcionando herramientas accesibles y poderosas para una amplia gama de aplicaciones de procesamiento del lenguaje natural. La sección "Models" de su sitio web es un recurso valioso para cualquier persona que busque trabajar con modelos de lenguaje preentrenados y de vanguardia.

---

Una cosa es un modelo y otra cosa diferente es una aplicación que yo vaya a montar usando ese modelo.

Chatbot -> Aplicación
Y para ese chatbot quizás voy a usar varios modelos.

Por un lado, voy a necesitar entender el contexto del que me está hablando el usuario.
Para eso usamos modelos de lenguaje de gran tamaño (LLM) pero que nos ayuden a clasificar el texto que nos está llegando en función de la intención del usuario / datos que nos proporciona el usuario.
Si el usuario me pide información de un PRODUCTO X -> Necesito un modelo que sepa identificar que me está hablando de un producto y que además me diga cuál es ese producto.
Lo mismo sería para un servicio.

$ En que puedo ayudarle? 
# Quiero información sobre el producto X -> [PRODUCTO X] CONTEXTO
    o
# Quiero pagar mis facturas pendientes

$ Qué información necesitas de ese producto? 
# Como se soluciona un error que tengo? -> MODELO BERT (CONTEXTO)
                                                        ^ Es un  texto gigantesco que contiene la respuesta a todas las preguntas que se le pueden hacer a un producto.

---

PySpark es una librería que tenemos en python para trabajar contra lo que llamamos un CLUSTER DE SPARK
De hecho hay muchas librerías para atacar a un cluster de Spark: Scala, Java, Python, R, etc.

Spark es un framework de computación distribuida que nos permite trabajar con grandes volúmenes de datos sobre un cluster de Hadoop.

Hadoop: Es una herramienta de software que permite al instalarse sobre un conjunto de máquinas, que esas máquinas trabajen en paralelo para resolver un problema. ESTA ES LA BASE DEL BIGDATA!

# BIGDATA

No es análisis de datos, no es inteligencia artificial, no es machine learning, no es deep learning, no es nada de eso.
Es procesamiento de datos a gran escala:
- Se generan muy rápido
- Son muchos
- Son muy complejos

Y el procesamiento podrá ser el que sea:
- Almacenamiento
- Transmitirlos
- Analizarlos

LA solución (comenzó GOOGLE con ello - BIGTABLE) fue: usar en lugar de una computadora gigante, un montón de ordenadores de mierda (commmodity hardware) y que trabajen en paralelo.

Y para ello crean:
- Un modelo de programación paralelizable llamado MAP-REDUCE
- Un sistema de archivos distribuido llamado GDFS (Google Distributed File System)

De aquello surge una implementación OPENSOURCE, llamada HADOOP, que es un conjunto de herramientas que nos permiten trabajar con un cluster de computadoras para resolver problemas de BIGDATA.

Hadoop nos ofrece:
- Una implementación del modelo de programación MAP-REDUCE
- Un sistema de archivos distribuido llamado HDFS (Hadoop Distributed File System)

El problema con Hadoop es que la implementación que hace de MapReduce es muy lenta (todas las operaciones hacen grabaciones en HDD intermedias). Y por eso surge SPARK (que es una reimplementación del MAP-REDUCE de Hadoop pero en memoria RAM)

La idea detrás de MapReduce es que tengo un volumen de datos ENORME (RDD) y lo que hago es dividirlo en trozos (particiones) y cada partición la proceso en un nodo del cluster (en paralelo), para que allí se procese y luego junto los resultados de cada nodo para obtener el resultado final.

de DAVID FERNANDEZ ROIBAS a todo el mundo:
    Lo mismo nos pasa en pyspark con la arquitectura que tenemos... mucho movimiento de datos pero poca chicha procesando.

Spark se base en cálculos en CPU... por muchas CPUs que tenga... nunca van a ir a la velocidad de las GPUs.
Hay que estar moviendo los datos de unas máquinas a otras... y eso es muy lento (Se manda por red)

Spark tiene una librería especializada en ML que se llama MLlib. Esa librería hace uso del MAP-REDUCE de Spark para hacer los cálculos de ML.
1- Los calculos van a CPU
2- Hay que mover los datos

¿Cuándo tiene sentido? Cuando ya tengo los datos en esos nodos. Si tengo que moverlos, no tiene sentido.
Y ahí entra el HDFS.

Yo voy a tener datos que me llegan ... a lo largo del tiempo y los voy guardando a lo largo del cluster.
Una parte de los datos en un nodo... otra parte de los datos en otro nodo... y así sucesivamente.
Y en un momento dado me planteo hacer un análisis de esos datos... 
Y lanzo ese proceso con Spark... y cada nodo va leyendo los datos que tiene y los procesa... 
AQUÍ ES DONDE TENGO UN GRAN RESULTADO CON SPARK y MLlib.... teniendo en cuenta la limitación de procesamiento de CPU

Como tenga que mover datos entre nodos... apaga y vamonos... Tengo 800 máquinas y las tengo aburridas, esperando  a que los datose vayan llegando o saliendo por la red.

---

# DEV-->OPS

Es  una cultura, un movimiento, una filosofía (NO ES un perfil profesional, no es una metodología)
en pro de LA AUTOMATIZACION de todo lo que hay entre DEV -> OPS


Es una evolución d lo que antes llamábamos SDMC

Una cosa es un proyecto de software... que antiguamente un proyecto lo gestionábamos mediante una metodología en cascada. Versión 1 de un programa que me han pedido.

Luego tendré una segunda versión... Sacaba un nuevo proyecto... que también gestionaba mediante una metodología en cascada.

Al mirar el ciclo de vida del software (la acumulación de esos proyectos) ---> SDMC