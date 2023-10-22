Proyecto creado en Intellj IDEA, para usar aprendizaje profundo para java: DeepLearning4Java o DL4J.
Se utilizo Java con la arquitectura de un proyecto Maven para importar dependencias para entrar la CNN. Estas dependencias se encuentran en el archivo pom.xml
La CNN fue entrenada con datos del MNIST(conjunto de imagenes de números de un dígito en escala de grises con ruido)
El modelo una vez entrenado se exporta en un zip para probarlo en una clase NumberRecognition, donde se tiene que cargar una imagen en escala de grises de tamaño 28x28 pixeles para que el modelo prediga.
