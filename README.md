# :smiley: Análisis de tweets de felicidad

El [corpus](https://github.com/GIL-UNAM/TwitterHappiness/blob/main/Dataset.csv) para este proyecto es una recopilación de 10048 tuits obtenidos de la búsqueda del tag #felicidad. El corpus recopilado fue dado a 3 voluntarios a quienes se les pidió etiquetaran cada tuit, según su criterio, en los que expresaran **alegría (A)**, **publicidad (P)**, **felicitaciones (F)**, **consejos (C)** y **no alegría o sarcasmos (N)**. Al finalizar el etiquetado se realizó un filtro para obtener aquellos tuits que coincidian en más de una etiqueta y aquellos en los que ocurrió lo contrario se clasificaron en una sexta categoría nombrada **No Agreement (NA)**. Por último, se realizó un preprocesamiento al corpus tokenizando, eliminando signos de puntuación e hiperenlaces además de una extracción de raíces. Lo anterior descrito puede ser encontrado en el archivo [Pre-procesamiento](https://github.com/GIL-UNAM/TwitterHappiness/blob/main/Pre-procesamiento.py).

Como parte del análisis, en el archivo [Frecuencias Relativas](https://github.com/GIL-UNAM/TwitterHappiness/blob/main/Frecuencias%20Relativas.py), se encuentra el  código para obtener las frecuencias de las palabras dentro de cada categoría y dentro del corpus total además de las frecuencias relativas de cada categoría con respecto al corpus total.

Por último, dentro del archivo [Sistemas de aprendizaje](https://github.com/GIL-UNAM/TwitterHappiness/blob/main/Sistemas%20de%20aprendizaje.py), se muestra el código de como se han aplicado los sistemas de aprendizaje **Naive Bayes (NB)**, **Logistic Regression (LR)**, **Random Forest (RF)** y **Support Vector Machine (SVM)** con conjuntos de <em>train-tests</em> en estratificaciones de 3 capas y obteniendo un porcentaje de exactitud y un score para cada sistema de aprendizaje en cada capa.

La lista de los principales paquetes empleados en la ejecución de los códigos se pueden encontrar en el archivo [Pre-requisitos](https://github.com/GIL-UNAM/TwitterHappiness/blob/main/Pre-requisitos.md).

## :pencil: Cómo citar

## Colaboradores
- Gemma Bel-Enguix, Instituto de Ingeniería - UNAM
- Helena Gómez Adorno, Instituto de Investigaciones en Matemáticas Aplicadas y en Sistemas - UNAM
- Karla Mendoza Grageda, Facultad de Ciencias - UNAM
- Grigori Sidorov, Instituto Politécnico Nacional - UNAM
