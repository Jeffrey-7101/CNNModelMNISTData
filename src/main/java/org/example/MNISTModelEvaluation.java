package org.example;

import org.deeplearning4j.datasets.iterator.*;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;

import javax.swing.*;
import java.awt.image.BufferedImage;
import java.io.File;

public class MNISTModelEvaluation {
    public static void main(String[] args) throws Exception {
        // Cargar el modelo previamente entrenado
        MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(new File("src/main/modelTrained/model.zip"));

        // Preparar un subconjunto del conjunto de datos MNIST para evaluación
        int batchSize = 64;
        DataSetIterator mnistTest = new MnistDataSetIterator(batchSize, false, 12345); // 12345 es una semilla aleatoria

        // Realizar la evaluación del modelo en el subconjunto de prueba MNIST
        Evaluation evaluation = new Evaluation(10); // 10 clases en MNIST (números del 0 al 9)
        while (mnistTest.hasNext()) {
            DataSet next = mnistTest.next();
            INDArray output = model.output(next.getFeatures());
            evaluation.eval(next.getLabels(), output);

            // Obtener los números reales y predichos
            int realLabel = Nd4j.argMax(next.getLabels(), 1).getInt(0);
            int predictedLabel = Nd4j.argMax(output, 1).getInt(0);

            // Mostrar la imagen
            BufferedImage image = new BufferedImage(28, 28, BufferedImage.TYPE_BYTE_GRAY);
            for (int i = 0; i < 28; i++) {
                for (int j = 0; j < 28; j++) {
                    int pixelValue = (int) (next.getFeatures().getDouble(0, i * 28 + j) * 255.0);
                    int rgb = (pixelValue << 16) | (pixelValue << 8) | pixelValue;
                    image.setRGB(i, j, rgb);
                }
            }
            ImageIcon icon = new ImageIcon(image);
            JOptionPane.showMessageDialog(null, "Número real: " + realLabel + "\nNúmero predicho: " + predictedLabel, "Imagen MNIST", JOptionPane.INFORMATION_MESSAGE, icon);
        }


        // Imprimir la precisión en el conjunto de prueba MNIST
        System.out.println("Precisión en el conjunto de prueba MNIST: " + evaluation.accuracy());
    }
}

